"""Synthetic metadata/callback tests; no media decoder or ML runtime is used."""

from contextlib import closing
import hashlib
import threading
import time
from types import SimpleNamespace as NS
import unittest

from serving.fast_encoder import EncodedBatch
from serving.feature_stream import EncodedSourceBatch, FeatureStreamError, iter_encoded_batches


def batch(index, *, total=3, views=1):
    content, pts = "1"*64, index*10
    frame = NS(content_hash=content, source_pts=pts, source_origin_pts=0,
        time_base_num=1, time_base_den=1000, pts_sec=pts/1000,
        next_source_pts=(index+1)*10 if index+1 < total else None,
        frame_id=hashlib.sha256(f"{content}:{pts}:1/1000".encode()).hexdigest())
    transforms = [NS(view_id="global", source_box_xyxy=(0,0,1,1))]
    if views == 2:
        transforms.append(NS(view_id="roi0", source_box_xyxy=(.1,.2,.8,.9)))
    return NS(frames=(frame,), transforms=tuple(transforms), shape=(views,256,256,3),
        pixels=b"private-generated-pixels", source_complete=index+1 == total,
        segment_complete=index+1 == total, tail_included=index+1 == total,
        decoded_source_frame_count=index+1, delivered_count=index+1,
        selection_mode="native", segment_id="private-source-segment")


class FakeEncoder:
    recipe_sha256 = "a"*64
    def __init__(self, callback=None):
        self.callback = callback
        self.calls = 0
    def encode_frame_batch(self, value, **kwargs):
        self.calls += 1
        if self.callback:
            self.callback(value, **kwargs)
        return EncodedBatch(object(), (value.shape[0],768), {"elapsed_sec":.001}, self.recipe_sha256)


class FeatureStreamTests(unittest.TestCase):
    def deadline(self):
        return time.monotonic()+3

    def test_complete_ordered_source_metadata_and_private_features_pass(self):
        encoder = FakeEncoder()
        output = list(iter_encoded_batches(lambda cancel: iter([batch(i) for i in range(3)]),
            encoder, deadline=self.deadline()))
        self.assertEqual(len(output), 3)
        self.assertEqual([value.frames[0].source_pts for value in output], [0,10,20])
        self.assertTrue(output[-1].source_complete and output[-1].tail_included)
        self.assertEqual(output[-1].decoded_source_frame_count, 3)
        self.assertNotIn("private", repr(output[0]))
        self.assertNotIn("features=", repr(output[0]))

    def test_decode_continues_while_encoding_with_bounded_lookahead(self):
        produced_second = threading.Event()
        first_encoding = threading.Event()
        produced = []
        def source(cancel):
            for index in range(6):
                produced.append(index)
                if index == 1:
                    produced_second.set()
                yield batch(index, total=6)
        def encode(value, **kwargs):
            if value.frames[0].source_pts == 0:
                first_encoding.set()
                self.assertTrue(produced_second.wait(1), "decoder did not overlap GPU consumer")
                time.sleep(.05)
                # Two lookahead reservations plus one active encoding batch.
                self.assertLessEqual(len(produced), 3)
        values = list(iter_encoded_batches(source, FakeEncoder(encode), deadline=self.deadline()))
        self.assertEqual(len(values), 6)
        self.assertTrue(first_encoding.is_set())

    def test_early_close_cancels_and_closes_reader_in_its_own_thread(self):
        closed = threading.Event()
        thread_names = []
        def source(cancel):
            try:
                for index in range(30):
                    yield batch(index, total=30)
            finally:
                thread_names.append(threading.current_thread().name)
                closed.set()
        with closing(iter_encoded_batches(source, FakeEncoder(), deadline=self.deadline())) as stream:
            next(stream)
        self.assertTrue(closed.is_set())
        self.assertEqual(thread_names, ["forte-feature-decode"])

    def test_decoder_error_text_is_never_forwarded(self):
        def source(cancel):
            raise RuntimeError("private-video-name-and-caption")
            yield
        with self.assertRaises(FeatureStreamError) as caught:
            list(iter_encoded_batches(source, FakeEncoder(), deadline=self.deadline()))
        self.assertEqual(str(caught.exception), "feature_stream_decoder_failed")

    def test_gap_and_false_completion_are_rejected(self):
        broken = batch(1)
        broken.frames[0].source_pts = 11
        for values in ([batch(0),broken], [batch(0)]):
            with self.subTest(count=len(values)), self.assertRaises(FeatureStreamError):
                list(iter_encoded_batches(lambda cancel: iter(values), FakeEncoder(), deadline=self.deadline()))
        false_eof = batch(0)
        false_eof.source_complete = True
        false_eof.tail_included = True
        with self.assertRaisesRegex(FeatureStreamError, "false_eof"):
            list(iter_encoded_batches(lambda cancel: iter([false_eof]), FakeEncoder(), deadline=self.deadline()))

    def test_deadline_stops_waiting_decoder_and_drains_thread(self):
        closed = threading.Event()
        def source(cancel):
            try:
                while not cancel.is_set():
                    time.sleep(.005)
                return
                yield
            finally:
                closed.set()
        with self.assertRaisesRegex(FeatureStreamError, "deadline"):
            list(iter_encoded_batches(source, FakeEncoder(), deadline=time.monotonic()+.05))
        self.assertTrue(closed.is_set())

    def test_encoder_failure_closes_producer_without_publishing_eof(self):
        closed = threading.Event()
        def source(cancel):
            try:
                for index in range(3):
                    yield batch(index)
            finally:
                closed.set()
        def fail(value, **kwargs):
            raise FeatureStreamError("synthetic_encoder_failure")
        with self.assertRaisesRegex(FeatureStreamError, "synthetic_encoder_failure"):
            list(iter_encoded_batches(source, FakeEncoder(fail), deadline=self.deadline()))
        self.assertTrue(closed.is_set())

    def test_roi_is_additional_global_view_and_bound_count(self):
        good = batch(0, total=1, views=2)
        output = list(iter_encoded_batches(lambda cancel: iter([good]), FakeEncoder(), deadline=self.deadline()))
        self.assertEqual(output[0].views_per_frame, 2)
        bad = batch(0, total=1)
        bad.transforms[0].source_box_xyxy = (.1,.1,.9,.9)
        with self.assertRaisesRegex(FeatureStreamError, "global_view_cropped"):
            list(iter_encoded_batches(lambda cancel: iter([bad]), FakeEncoder(), deadline=self.deadline()))

    def test_unit_timebase_delegates_to_the_decoder_identity_protocol(self):
        from forte.video.batches import source_frame_id
        value = batch(0, total=1)
        frame = value.frames[0]
        frame.time_base_num, frame.time_base_den = 1, 1
        frame.frame_id = source_frame_id(frame.content_hash, frame.source_pts, 1, 1)
        output = list(iter_encoded_batches(lambda cancel: iter([value]), FakeEncoder(), deadline=self.deadline()))
        self.assertEqual(output[0].frames[0].frame_id, frame.frame_id)


if __name__ == "__main__":
    unittest.main()

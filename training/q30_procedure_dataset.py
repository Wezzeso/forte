"""Generate synthetic structured procedure timelines; never reads image/video/labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


TYPES = ["NO_RECOUNT", "KEY_LEFT_IN_SAFE", "COUNT_WITHOUT_CLIENT", "NO_CLIENT_ID",
         "NO_AUTHENTICITY_CHECK", "CASH_LEFT_UNATTENDED", "COLLECTOR_NO_DOCUMENTS",
         "CASH_HANDOVER_IRREGULAR", "WORKSTATION_UNLOCKED"]
SCHEMA = {"type": "object", "properties": {
    "type_id": {"type": "string", "enum": TYPES},
    "decision": {"type": "string", "enum": ["supported", "not_supported", "unassessable"]},
    "anchor_index": {"type": "integer", "minimum": -1}},
    "required": ["type_id", "decision", "anchor_index"], "additionalProperties": False}
SYSTEM = (
    "You evaluate one requested visible cash-procedure check from a structured synthetic event timeline. "
    "Treat only observed=true events as evidence. Object tokens refer only to local tracked objects; "
    "never infer identities, text, amounts or invisible actions. Return strictly the three-field JSON schema. "
    "supported requires positive evidence for the requested type. not_supported means the clearly observed "
    "near-neighbor sequence contradicts that type. unassessable means decisive coverage or observations are unknown. "
    "For missing recount, identity/document check, authenticity check, or incomplete handover, require the complete "
    "relevant transaction, observed_fraction>=0.90 and all critical regions visible; unknown never means omitted. "
    "For a key left in a safe require the same continuously tracked key and lock after hand withdrawal and "
    "observed cashier departure; an earlier key insertion is insufficient. COUNT_WITHOUT_CLIENT needs a visible "
    "counting interval and continuously visible empty client region. CASH_LEFT_UNATTENDED needs visible cash "
    "remaining after departure and a continuously observed absence of a responsible person. WORKSTATION_UNLOCKED "
    "needs visible working screen content after cashier departure; an unseen screen is unknown. "
    "Use a zero-based timeline index for supported, anchoring at the decisive observed event. "
    "For not_supported or unassessable anchor_index is -1. Evaluate only requested_type."
    " COUNT_WITHOUT_CLIENT additionally requires a visually linked unfinished client-cash operation; counting "
    "cash taken from and returned to a safe without a client link is not that check. An unused separate detector "
    "does not prove missing authenticity verification when a machine-integrated route is unknown: return "
    "unassessable unless all used authentication routes are visibly resolved. Mere partial cash retention "
    "does not prove handover irregularity, amount, ownership or intent. Require a tracked visible offer of the "
    "same cash group, subsequent partial transfer and withdrawal of the visibly offered remainder; otherwise "
    "return unassessable. These are visible advisory signs, not legal conclusions."
)
USER_PREFIX = "Evaluate the requested check using only this structured event timeline:\n"

# Forty independent scenario templates; variants inherit their template split.
# Each tuple: type, Russian scenario, supportive event sequence, correct near-neighbor.
TEMPLATES = [
 ("NO_RECOUNT", "Приём отдельных банкнот и помещение в сейф", "cash_accepted,banknotes_arranged,cash_stored", "cash_accepted,banknotes_arranged,machine_count_completed,cash_stored"),
 ("NO_RECOUNT", "Вскрытие инкассаторского мешка перед хранением", "bag_opened,bundles_removed,cash_stored", "bag_opened,bundles_removed,manual_count_completed,cash_stored"),
 ("NO_RECOUNT", "Перенос банкнот с лотка в кассету сейфа", "cash_accepted,cash_on_tray,cashette_loaded,cash_stored", "cash_accepted,cash_on_tray,machine_count_completed,cashette_loaded,cash_stored"),
 ("NO_RECOUNT", "Упаковка принятых банкнот в конверт", "cash_accepted,envelope_loaded,cash_stored", "cash_accepted,manual_count_completed,envelope_loaded,cash_stored"),
 ("NO_RECOUNT", "Несколько пачек после вскрытия упаковки", "package_opened,bundles_separated,all_bundles_stored", "package_opened,bundles_separated,all_bundles_count_completed,all_bundles_stored"),
 ("NO_RECOUNT", "Промежуточный лоток перед помещением в сейф", "cash_accepted,cash_moved_to_staging_tray,cash_stored", "cash_accepted,cash_moved_to_staging_tray,manual_count_completed,cash_stored"),
 ("KEY_LEFT_IN_SAFE", "Закрытая дверь сейфа, кассир ушёл от сейфа", "key_inserted,safe_opened,safe_closed,hand_withdrawn,cashier_left_safe_zone,key_visible_after_departure", "key_inserted,safe_opened,safe_closed,key_removed,hand_withdrawn,cashier_left_safe_zone,lock_observed_empty"),
 ("KEY_LEFT_IN_SAFE", "Ключ остаётся после выхода кассира из комнаты", "key_inserted,safe_closed,hand_withdrawn,cashier_left_room,key_visible_after_departure", "key_inserted,safe_closed,key_removed,hand_withdrawn,cashier_left_room,lock_observed_empty"),
 ("KEY_LEFT_IN_SAFE", "Повторное закрытие сейфа после второй операции", "key_inserted,safe_opened,safe_closed,safe_reopened,safe_closed,hand_withdrawn,cashier_left_safe_zone,key_visible_after_departure", "key_inserted,safe_opened,safe_closed,safe_reopened,safe_closed,key_removed,hand_withdrawn,cashier_left_safe_zone,lock_observed_empty"),
 ("KEY_LEFT_IN_SAFE", "Кассир вернулся к столу после сейфа", "key_inserted,safe_closed,hand_withdrawn,cashier_returned_to_desk,key_visible_after_departure", "key_inserted,safe_closed,key_removed,hand_withdrawn,cashier_returned_to_desk,lock_observed_empty"),
 ("KEY_LEFT_IN_SAFE", "Уборка лотка и отход от сейфа", "key_inserted,safe_closed,tray_removed,hand_withdrawn,cashier_left_safe_zone,key_visible_after_departure", "key_inserted,safe_closed,key_removed,tray_removed,hand_withdrawn,cashier_left_safe_zone,lock_observed_empty"),
 ("KEY_LEFT_IN_SAFE", "Окончание работы у сейфа с контролем замка", "safe_opened,key_inserted,safe_closed,hand_withdrawn,cashier_left_safe_zone,key_visible_after_departure", "safe_opened,key_inserted,safe_closed,key_removed,hand_withdrawn,cashier_left_safe_zone,lock_observed_empty"),
 ("COUNT_WITHOUT_CLIENT", "Пересчёт начинается после ухода клиента", "client_left,client_region_observed_empty,count_started,count_completed", "client_present,count_started,count_completed,client_left"),
 ("COUNT_WITHOUT_CLIENT", "Ручной пересчёт без клиента у стола", "client_left,client_region_observed_empty,manual_count_started,count_completed", "client_present,manual_count_started,count_completed,client_left"),
 ("COUNT_WITHOUT_CLIENT", "Клиент отсутствует весь машинный пересчёт", "client_region_observed_empty,cash_loaded_in_counter,count_started,count_completed", "client_present,cash_loaded_in_counter,count_started,count_completed"),
 ("COUNT_WITHOUT_CLIENT", "Незавершённая операция клиента против пересчёта из сейфа", "client_present,initial_count_completed,client_left,client_region_observed_empty,count_started,count_completed", "cash_removed_from_safe_for_count,client_region_observed_empty,count_started,count_completed,cash_returned_to_same_safe"),
 ("NO_CLIENT_ID", "Передача наличных без предъявления документа", "client_present,cash_accepted,service_completed", "client_present,document_presented,document_checked,cash_accepted,service_completed"),
 ("NO_CLIENT_ID", "Внесение наличных после посадки клиента", "client_sat,cash_accepted,cash_processed,service_completed", "client_sat,document_presented,document_checked,cash_accepted,cash_processed,service_completed"),
 ("NO_CLIENT_ID", "Приём конверта с деньгами от клиента", "client_present,envelope_opened,cash_accepted,service_completed", "client_present,document_presented,document_checked,envelope_opened,cash_accepted,service_completed"),
 ("NO_CLIENT_ID", "Вторая операция нового клиента", "previous_client_left,new_client_present,cash_accepted,service_completed", "previous_client_left,new_client_present,document_presented,document_checked,cash_accepted,service_completed"),
 ("NO_AUTHENTICITY_CHECK", "Пересчитанные банкноты сразу убраны", "cash_accepted,machine_count_completed,cash_stored", "cash_accepted,machine_count_completed,authenticity_check_completed,cash_stored"),
 ("NO_AUTHENTICITY_CHECK", "Ручной пересчёт перед хранением", "cash_accepted,manual_count_completed,cash_stored", "cash_accepted,manual_count_completed,authenticity_check_completed,cash_stored"),
 ("NO_AUTHENTICITY_CHECK", "Обработка нескольких пачек", "bundles_removed,all_bundles_count_completed,all_bundles_stored", "bundles_removed,all_bundles_count_completed,all_bundles_authenticity_checked,all_bundles_stored"),
 ("NO_AUTHENTICITY_CHECK", "Закрытие операции после приёма наличных", "cash_accepted,count_completed,service_completed", "cash_accepted,count_completed,authenticity_check_completed,service_completed"),
 ("CASH_LEFT_UNATTENDED", "Наличные на столе после выхода кассира", "cash_on_desk,cashier_left_room,responsible_region_observed_empty,cash_visible_after_departure", "cash_on_desk,cash_secured,cashier_left_room,desk_observed_clear"),
 ("CASH_LEFT_UNATTENDED", "Кассир уходит, деньги остаются на лотке", "cash_on_tray,cashier_left_desk,responsible_region_observed_empty,cash_visible_after_departure", "cash_on_tray,cash_secured,cashier_left_desk,desk_observed_clear"),
 ("CASH_LEFT_UNATTENDED", "Пачка у счётной машины без ответственного", "cash_beside_counter,cashier_left_room,responsible_region_observed_empty,cash_visible_after_departure", "cash_beside_counter,cash_secured,cashier_left_room,desk_observed_clear"),
 ("CASH_LEFT_UNATTENDED", "После клиента кассир покидает рабочую зону", "client_left,cash_on_desk,cashier_left_desk,responsible_region_observed_empty,cash_visible_after_departure", "client_left,cash_on_desk,cash_secured,cashier_left_desk,desk_observed_clear"),
 ("COLLECTOR_NO_DOCUMENTS", "Приём мешка инкассатора без документов", "collector_entered,bag_presented,bag_accepted,collector_service_completed", "collector_entered,collector_document_presented,collector_document_checked,bag_presented,bag_accepted,collector_service_completed"),
 ("COLLECTOR_NO_DOCUMENTS", "Сумка передана после подхода к столу", "collector_at_desk,bag_presented,bag_accepted,collector_service_completed", "collector_at_desk,collector_document_presented,collector_document_checked,bag_presented,bag_accepted,collector_service_completed"),
 ("COLLECTOR_NO_DOCUMENTS", "Приём второй сумки в отдельном визите", "previous_collector_left,collector_entered,bag_presented,bag_accepted,collector_service_completed", "previous_collector_left,collector_entered,collector_document_presented,collector_document_checked,bag_presented,bag_accepted,collector_service_completed"),
 ("COLLECTOR_NO_DOCUMENTS", "Открытие принятого мешка", "collector_entered,bag_accepted,bag_opened,collector_service_completed", "collector_entered,collector_document_presented,collector_document_checked,bag_accepted,bag_opened,collector_service_completed"),
 ("CASH_HANDOVER_IRREGULAR", "После разделения часть банкнот удержана", "cash_prepared,cash_split,part_handed_to_client,part_retained,client_left,service_ended_with_retained_cash", "cash_prepared,cash_split,part_handed_to_client,remaining_part_handed_to_client,client_left,service_completed"),
 ("CASH_HANDOVER_IRREGULAR", "Часть суммы остаётся в лотке до ухода клиента", "cash_on_tray,cash_split,part_handed_to_client,part_retained_on_tray,client_left,service_ended_with_retained_cash", "cash_on_tray,cash_split,part_handed_to_client,remaining_part_handed_to_client,client_left,service_completed"),
 ("CASH_HANDOVER_IRREGULAR", "После пересчёта передана только одна часть", "count_completed,cash_split,part_handed_to_client,part_retained,client_left,service_ended_with_retained_cash", "count_completed,cash_split,part_handed_to_client,remaining_part_handed_to_client,client_left,service_completed"),
 ("CASH_HANDOVER_IRREGULAR", "Передача двух пачек завершается удержанием одной", "bundles_prepared,cash_split,part_handed_to_client,part_retained,client_left,service_ended_with_retained_cash", "bundles_prepared,cash_split,part_handed_to_client,remaining_part_handed_to_client,client_left,service_completed"),
 ("WORKSTATION_UNLOCKED", "Рабочее приложение после выхода из комнаты", "working_screen_visible,cashier_left_room,working_screen_visible_after_departure", "working_screen_visible,screen_locked,cashier_left_room,lock_screen_visible_after_departure"),
 ("WORKSTATION_UNLOCKED", "Отход от стола при рабочем экране", "working_screen_visible,cashier_left_desk,working_screen_visible_after_departure", "working_screen_visible,screen_locked,cashier_left_desk,lock_screen_visible_after_departure"),
 ("WORKSTATION_UNLOCKED", "Уход после окончания обслуживания", "service_completed,working_screen_visible,cashier_left_room,working_screen_visible_after_departure", "service_completed,working_screen_visible,screen_locked,cashier_left_room,lock_screen_visible_after_departure"),
 ("WORKSTATION_UNLOCKED", "Рабочий экран остаётся после возврата к сейфу", "working_screen_visible,cashier_left_desk,cashier_at_safe,working_screen_visible_after_departure", "working_screen_visible,screen_locked,cashier_left_desk,cashier_at_safe,lock_screen_visible_after_departure"),
]

ANCHORS = {"NO_RECOUNT": {"cash_stored", "all_bundles_stored"},
           "KEY_LEFT_IN_SAFE": {"key_visible_after_departure"},
           "COUNT_WITHOUT_CLIENT": {"count_started", "manual_count_started"},
           "NO_CLIENT_ID": {"cash_accepted"},
           "NO_AUTHENTICITY_CHECK": {"cash_stored", "all_bundles_stored", "service_completed"},
           "CASH_LEFT_UNATTENDED": {"cash_visible_after_departure"},
           "COLLECTOR_NO_DOCUMENTS": {"bag_accepted"},
           "CASH_HANDOVER_IRREGULAR": {"service_ended_with_retained_cash"},
           "WORKSTATION_UNLOCKED": {"working_screen_visible_after_departure"}}
CONTROLS = {"seed": 20261018, "template_groups": 40, "train_groups": 32,
            "validation_groups": 8, "variants_per_group": 16,
            "variant_modes": ["supported", "correct_neighbor", "incomplete_or_track_unknown", "critical_region_unknown"],
            "event_spacing_seconds": [0.4, 0.8, 1.2, 1.6],
            "time_offset_seconds": [0.0, 2.0, 7.0], "object_token_variants": 100,
            "neutral_distractor_count": [0, 1], "high_observed_fraction": [0.95, 1.0],
            "low_observed_fraction": [0.60, 0.89], "split_rule": "seeded_template_hash_32_8",
            "anchor_rule": "decisive_event_index_else_minus_one"}


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def validate_semantics(row: dict) -> None:
    value, target = row["input"], row["target"]
    kind, coverage, timeline = value["requested_type"], value["coverage"], value["timeline"]
    assert kind in TYPES and target["type_id"] == kind
    assert target["decision"] in {"supported", "not_supported", "unassessable"}
    assert set(value) == {"requested_type", "coverage", "timeline"}
    assert row["origin"] == "synthetic" and row["modality"] == "text"
    assert all(timeline[i]["t_sec"] < timeline[i+1]["t_sec"] for i in range(len(timeline)-1))
    if target["decision"] != "supported":
        assert target["anchor_index"] == -1
        return
    anchor = target["anchor_index"]
    assert 0 <= anchor < len(timeline) and timeline[anchor]["observed"] is True
    assert timeline[anchor]["event"] in ANCHORS[kind]
    assert coverage["critical_regions_visible"] is True
    observed = {e["event"] for e in timeline if e["observed"]}
    if kind != "KEY_LEFT_IN_SAFE" and kind != "WORKSTATION_UNLOCKED":
        assert coverage["transaction_complete"] is True and coverage["observed_fraction"] >= .90
        assert "transaction_completed" in observed
    if kind == "KEY_LEFT_IN_SAFE":
        assert all(coverage[k] is True for k in ("same_key_track", "same_lock_track",
                   "hand_withdrawal_observed", "cashier_departure_observed"))
        assert {"hand_withdrawn", "key_visible_after_departure"} <= observed
        assert observed & {"cashier_left_safe_zone", "cashier_left_room", "cashier_returned_to_desk"}
    if kind == "COUNT_WITHOUT_CLIENT":
        assert coverage["client_cash_link_visible"] and coverage["client_region_continuously_visible"]
        assert "cash_accepted_from_client" in observed and "client_cash_operation_open" in observed
    if kind == "NO_AUTHENTICITY_CHECK":
        assert coverage["authentication_route_known"] is True
    if kind == "CASH_LEFT_UNATTENDED":
        assert coverage["responsible_region_continuously_visible"] is True
        assert "responsible_region_observed_empty" in observed
    if kind == "CASH_HANDOVER_IRREGULAR":
        assert coverage["offered_cash_group_track_visible"] is True
        assert {"whole_cash_group_visibly_offered", "client_reached_for_offered_group",
                "part_handed_to_client", "offered_remainder_withdrawn_to_cashier_side"} <= observed


def build_row(group: int, template: tuple, split: str, variant: int) -> dict:
    type_id, _, positive, correct = template
    rng = random.Random(sha([CONTROLS["seed"], group, variant]))
    mode = variant % 4
    sequence = (correct if mode == 1 else positive).split(",")
    if type_id == "COUNT_WITHOUT_CLIENT" and not (mode == 1 and "cash_removed_from_safe_for_count" in sequence):
        sequence = ["cash_accepted_from_client", "client_cash_operation_open"] + sequence
    if type_id == "CASH_HANDOVER_IRREGULAR" and mode != 1:
        sequence = ["whole_cash_group_visibly_offered", "client_reached_for_offered_group"] + sequence
        sequence = ["offered_remainder_withdrawn_to_cashier_side" if e in {"part_retained", "part_retained_on_tray"} else e for e in sequence]
    object_token, key_token, lock_token = ["o" + str(value) for value in rng.sample(range(100), 3)]
    if rng.randrange(2):
        sequence.insert(0, "neutral_tray_repositioned")
    sequence = ["transaction_started"] + sequence + ["transaction_completed"]
    step, offset = rng.choice(CONTROLS["event_spacing_seconds"]), rng.choice(CONTROLS["time_offset_seconds"])
    events = []
    for i, event in enumerate(sequence):
        role = "collector" if "collector" in event or "bag_" in event else "client" if "client" in event else "cashier"
        object_id = key_token if "key" in event else lock_token if "safe" in event or "lock" in event else object_token
        events.append({"event": event, "role": role, "object_token": object_id,
                       "t_sec": round(offset + i * step, 3), "observed": True})
    coverage = {"transaction_complete": True, "observed_fraction": rng.choice([0.95, 1.0]),
                "critical_regions_visible": True, "same_cash_track": True,
                "same_key_track": True, "same_lock_track": True,
                "hand_withdrawal_observed": True, "cashier_departure_observed": True,
                "client_region_continuously_visible": True,
                "responsible_region_continuously_visible": True,
                "client_cash_link_visible": not (mode == 1 and "cash_removed_from_safe_for_count" in sequence),
                "authentication_route_known": True,
                "offered_cash_group_track_visible": True}
    if mode == 2:
        if type_id == "KEY_LEFT_IN_SAFE":
            coverage["same_key_track"] = False
            coverage["same_lock_track"] = False
        else:
            coverage["transaction_complete"] = False
            coverage["observed_fraction"] = rng.choice([0.60, 0.89])
            coverage["critical_regions_visible"] = False
        # An explicitly unobserved completion cannot prove complete coverage.
        events[-1]["observed"] = False
    if mode == 3:
        coverage["critical_regions_visible"] = False
        coverage["hand_withdrawal_observed"] = False if type_id == "KEY_LEFT_IN_SAFE" else True
        coverage["client_region_continuously_visible"] = False if type_id == "COUNT_WITHOUT_CLIENT" else True
        coverage["responsible_region_continuously_visible"] = False if type_id == "CASH_LEFT_UNATTENDED" else True
        if type_id == "NO_AUTHENTICITY_CHECK":
            coverage["authentication_route_known"] = False
        if type_id == "CASH_HANDOVER_IRREGULAR":
            coverage["offered_cash_group_track_visible"] = False
        if type_id == "COUNT_WITHOUT_CLIENT":
            coverage["client_cash_link_visible"] = False
        for event in events:
            if event["event"] in ANCHORS[type_id]:
                event["observed"] = False
    # A counting-machine path with unresolved integrated verification is unknown,
    # even with a fully observed unused separate detector. No omission is invented.
    unresolved_integrated_route = (type_id == "NO_AUTHENTICITY_CHECK" and mode != 1
                                  and "machine_count_completed" in sequence)
    if unresolved_integrated_route:
        coverage["authentication_route_known"] = False
    decision = "supported" if mode == 0 else "not_supported" if mode == 1 else "unassessable"
    if unresolved_integrated_route:
        decision = "unassessable"
    anchor = next((i for i, event in enumerate(events) if event["event"] in ANCHORS[type_id]), -1) if decision == "supported" else -1
    assert decision != "supported" or anchor >= 0
    return {"origin": "synthetic", "modality": "text", "split": split,
            "template_group": f"synthetic_procedure_{group:02d}",
            "variant_index": variant, "system": SYSTEM,
            "input": {"requested_type": type_id, "coverage": coverage, "timeline": events},
            "target": {"type_id": type_id, "decision": decision, "anchor_index": anchor}}


def generate(output: Path, docs: Path) -> dict:
    assert len(TEMPLATES) == 40 and not output.exists()
    order = sorted(range(40), key=lambda index: sha([CONTROLS["seed"], "split", index]))
    train_groups = set(order[:32])
    principle = "Видимое завершённое действие подтверждает шаг; отсутствие шага оценивается только при полной операции и наблюдении не менее 90%; неизвестное не является нарушением."
    lines = ["# Синтетический учебный набор процедур кассы — 18 октября 2026", "", principle, "",
             "Набор сгенерирован независимо от банковских видео, кадров, имён файлов и разметки. Конкретные пользовательские примеры не использованы. Это учебные логические последовательности, а не новые наблюдения или нормативные заключения.", "",
             "До эксперимента зафиксированы **14 управляющих параметров генерации** (список в receipt.json); ни один не выбран по результату модели. 40 групп шаблонов разделены по фиксированному seed: 32 обучающих и 8 проверочных; производные одной группы не переходят границу. На группу приходится 16 вариантов: 4 кандидата с полной видимостью, 4 корректных соседа, 4 с неполной операцией/неизвестным треком и 4 с невидимой критической областью. Кандидаты с неизвестным встроенным маршрутом проверки подлинности также имеют решение unassessable. Итого 512/128 примеров.", "",
             "Вход модели содержит только requested_type, coverage и typed timeline; имена групп, split, seed и номера вариантов в prompt не попадают. t_sec — относительное время событий, не длительность ролика. Локальные object_token обозначают синтетические треки, не идентичности людей. Все поля target — закрытые перечисления и индекс события.", "",
             "## Обязательные ограничения видимости", "",
             "- Для пропуска пересчёта, документа, проверки подлинности или завершения выдачи нужны полная операция, observed_fraction ≥ 0.90 и видимость критических зон. Невидимый или неизвестный шаг не считается пропущенным.",
             "- Для ключа нужны один и тот же замок и ключ, непрерывные треки, отведённая рука, наблюдаемый уход кассира и ключ после ухода. Одного прежнего вставления недостаточно.",
             "- Для пересчёта без клиента нужны видимая связь банкнот с незавершённой клиентской операцией и непрерывная видимость клиентской зоны. Пересчёт денег, взятых из сейфа и возвращённых туда без клиентской связи, сам по себе не подтверждает этот тип.",
             "- Неиспользованный отдельный детектор не доказывает пропуск проверки, если возможная встроенная проверка счётной машины неизвестна. Такой маршрут помечен unassessable. Для оставленных денег видны сами деньги и отсутствие ответственного после ухода.",
             "- Частичное удержание денег само по себе не доказывает сумму, собственность, намерение или неправильность выдачи. Подтверждающие синтетические варианты показывают предложение одной отслеженной группы клиенту, его видимый жест получения, передачу части и отвод предложенного остатка. Без этой наблюдаемой связи решение unassessable.",
             "- Для незаблокированного рабочего места рабочее содержимое экрана видно после ухода. Чёрный, закрытый или невидимый экран не доказывает нарушение.",
             "- Неподтверждённое отсутствие шага возвращает unassessable; anchor_index=-1. Для supported индекс указывает наблюдаемое решающее событие. not_supported также имеет индекс -1.", "",
             "## 40 конкретных пар последовательностей", "",
             "Английские названия ниже — закрытые типы событий, не свободные описания сцен. В каждой паре одинаковые токены объектов связаны внутри одной операции. Для каждой пары также создаются два вида недостаточной видимости, описанные выше.", ""]
    for i, (kind, title, positive, correct) in enumerate(TEMPLATES):
        split = "обучение" if i in train_groups else "проверка"
        lines.extend([f"{i+1}. **{title}** (`{kind}`, {split}).",
                      f"   Кандидат с обязательными защитными условиями: `{positive.replace(',', ' → ')}`.",
                      f"   Корректный сосед: `{correct.replace(',', ' → ')}`.", ""])
    lines.extend(["## Пределы результата", "",
                  "Это новый синтетический holdout по шаблонам, но не банковская оценка и не проверка зрения. Он не подтверждает качество восприятия пересчёта, ключей, документов или экранов на видео. Улучшение текстовой логики не является результатом 37/40. Синтетические примеры не заменяют отрицательный гейт на 19 банковских отрицательных роликах и не разрешают включение новых типов.", "",
                  "В наборе нет юридических цитат, аудио, лиц, личностей, сумм, OCR, кадров или организаторских меток. Допускаются только существующие catalog type_id. Выбор checkpoint по банковской разметке запрещён.", ""])
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n".join(lines), encoding="utf-8")
    output.mkdir(parents=True)
    rows = [build_row(i+1, template, "train" if i in train_groups else "validation", v)
            for i, template in enumerate(TEMPLATES) for v in range(16)]
    for row in rows:
        validate_semantics(row)
    hashes = {}
    for name in ("train", "validation", "manifest"):
        selected = rows if name == "manifest" else [r for r in rows if r["split"] == name]
        path = output / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
                                for r in selected), encoding="utf-8", newline="\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    license_data = {"origin": "synthetic", "external_data_inputs": [],
                    "new_source_dataset_license_required": False,
                    "statement": "Generated independently from generic procedure definitions; no bank examples or labels; rights to project-generated outputs remain with the project/user; no license for pre-existing project source is invented.",
                    "model_license_separate": "Qwen3-VL-30B-A3B-Instruct Apache-2.0; model weights not included"}
    (output / "license-generated-data.json").write_text(json.dumps(license_data, indent=2)+"\n")
    receipt = {"schema_version": "forte.q30.synthetic-procedure.v1", "origin": "synthetic",
               "state": "prepared_untrained", "controls": CONTROLS, "generation_control_count": len(CONTROLS),
               "principle_before_experiment": principle, "manifest_sha256": hashes,
               "group_counts": {"train": 32, "validation": 8}, "example_counts": {"train": 512, "validation": 128},
               "system": SYSTEM, "user_prefix": USER_PREFIX, "input_serialization": "json.dumps(input,sort_keys=True,separators=(',',':'))",
               "output_schema": SCHEMA, "input_schema": {"requested_type": TYPES,
                   "coverage": "explicit boolean guards plus observed_fraction in [0,1]",
                   "timeline": "ordered list {event:string enum,role:cashier|client|collector,object_token:synthetic local token,t_sec:relative number,observed:boolean}"},
               "event_enum": sorted({event["event"] for row in rows for event in row["input"]["timeline"]}),
               "target_anchor_semantics": "supported:zero-based decisive observed event index; otherwise -1",
               "bank_data_or_labels_or_images_read": False, "file_identifiers_in_model_input": False,
               "synthetic_validation_not_bank_score": True,
               "license_generated_data_sha256": hashlib.sha256((output / "license-generated-data.json").read_bytes()).hexdigest(),
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "russian_protocol_sha256": hashlib.sha256(docs.read_bytes()).hexdigest()}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("training/private/q30-procedure-v1"))
    p.add_argument("--docs", type=Path, default=Path("docs/integration/procedure-curriculum-20261018.md"))
    a = p.parse_args()
    print(json.dumps(generate(a.output, a.docs), ensure_ascii=False))

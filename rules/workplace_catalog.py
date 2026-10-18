"""Explicit optional workplace profile; never selected by the production loader.

The raw schema is intentionally incompatible with ``load_catalog``. New fact
axes require new observations/calibration, not relabeling a nineteen-axis head.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from forte.shared.schemas import FactDefinition
from rules.catalog import CATALOG_PATH, RuleCatalog, load_catalog

PROFILE_PATH = Path(__file__).with_name("workplace_catalog.v1.json")
TAXONOMY_PATH = Path(__file__).parent.parent / "annotations/workplace-taxonomy.v1.json"
SCHEMA = "forte.optional.workplace.rules.v1"
VERSION = "2026-10-17.workplace.public.v1"
VOCABULARY_VERSION = "forte.optional.workplace.fact-axes.v1"
RULE_IDS = (
    "OPT_RK_NBK231_P12_PERSONAL_STORAGE",
    "OPT_RK_NBK231_P12_RECORDING_COMM_DEVICE",
    "OPT_RK_HEALTH110_5_9_WORKPLACE_TOBACCO",
)


def digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _fact(key: str, question: str, definition: str, phrase: str, absence: str | None = None):
    result = {"fact_id": key, "question_ru": question, "definition_ru": definition,
              "description_ru": phrase}
    if absence is not None:
        result["absence_description_ru"] = absence
    return result


def _new_facts():
    rows = [
        _fact("WP_CONFIRMED_BANK_CASH_NODE", "Подтвержден ли именно банковский кассовый узел?",
              "Требуются отдельная доверенная настройка bank_cash_node и видимая зона внутри кассового узла. Настройка bank_cash_workstation, логотип, стол, одежда или присутствие денег не доказывают эту границу. Операционный зал и зона клиента снаружи не подходят.",
              "подтверждена зона внутри банковского кассового узла"),
        _fact("WP_CASH_NODE_BOUNDARY_VISIBLE", "Надежно ли определена и видна граница кассового узла?",
              "Кадры и доверенная схема зоны позволяют отличить внутреннюю часть кассового узла от клиентского зала. Не считать весь кадр служебной зоной по умолчанию; неизвестная граница означает unclear.",
              "граница кассового узла надежно определена и видна"),
        _fact("WP_WORKER_ROLE_BY_CASH_ACTIONS", "Установлена ли роль работника по его наблюдаемым кассовым действиям?",
              "Тот же анонимно прослеживаемый человек непосредственно выполняет банковскую кассовую работу на стороне кассы. Одежда, возраст, лицо, бейдж, пол и нахождение за столом сами по себе роль не устанавливают. Личность и полномочия не определяются.",
              "роль работника подтверждена наблюдаемыми кассовыми действиями"),
        _fact("WP_PERSONAL_ITEM_PROVENANCE_CLEAR", "Подтверждено ли личное назначение именно этого предмета?",
              "Нужен положительный наблюдаемый контекст личного использования и непрерывная связь предмета с тем же работником. Не выводить собственность из расположения, кармана, цвета или вида сумки. Предмет клиента, кассета, банковское рабочее устройство и неизвестное содержимое сумки требуют absent либо unclear.",
              "личное назначение прослеженного предмета подтверждено наблюдаемым контекстом"),
        _fact("WP_PERSONAL_STORAGE_EPISODE_VISIBLE", "Виден ли законченный эпизод хранения личного предмета внутри узла?",
              "Виден переход того же предмета в место хранения и его оставление там после завершения размещения. Просто носимая одежда, краткий перенос, процесс кассовой обработки и неизвестная дальнейшая судьба предмета не доказывают хранение.",
              "прослеженный личный предмет оставлен на хранение внутри кассового узла"),
        _fact("WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM", "Является ли предмет лекарством, медицинским изделием или средством связи без камеры?",
              "Проверяются лекарства, медицинские изделия и средства связи без функции фото/видеофиксации. absent допустимо только если сам предмет достаточно различим и его принадлежность к этим категориям исключена положительными признаками. Скрытое содержимое сумки, неизвестный прибор или неразличимая камера означают unclear, а не absent.",
              "предмет относится к разрешенным лекарствам, медицинским изделиям или средствам связи без камеры",
              "по различимым признакам предмет не относится к разрешенным медицинским изделиям, лекарствам или средствам связи без камеры"),
        _fact("WP_WORKER_ITEM_TRACK_CONTINUOUS", "Прослежены ли тот же работник и тот же предмет без разрыва?",
              "Исходные кадры связывают наблюдаемую кассовую роль, личный контекст предмета и рассматриваемое размещение либо переход границы. Подмена, заслонение с возможной подменой и выход из кадра требуют unclear. Не распознавать лица. Отдельные несвязанные предметы и люди не объединяются.",
              "тот же работник и предмет непрерывно прослежены через необходимые действия"),
        _fact("WP_COMM_DEVICE_IDENTIFIABLE", "Различимо ли именно переносимое средство связи?",
              "Нужно отличить конкретное переносимое коммуникационное устройство от калькулятора, кошелька, блокнота, медицинского прибора и неподвижной банковской системы видеонаблюдения. Прямоугольная форма, касание экрана или жест у лица недостаточны.",
              "прослеженный предмет различим как переносимое средство связи"),
        _fact("WP_RECORDING_CAPABILITY_VISIBLE", "Подтверждена ли функция фото или видео у этого устройства?",
              "Требуется ясно различимая камера/ее функция у того же средства связи либо наблюдаемое реальное действие съемки. Не предполагать камеру у любого телефона, прямоугольника или темного пятна. Содержимое личной переписки и изображений не читать и не раскрывать. Неизвестная функция означает unclear.",
              "у того же средства связи наблюдаемо подтверждена функция фото- или видеофиксации"),
        _fact("WP_DEVICE_ENTRY_INTO_NODE_VISIBLE", "Виден ли внос того же устройства работником внутрь кассового узла?",
              "Непрерывная последовательность показывает перенос работником того же устройства снаружи внутрь надежно определенной границы. Уже лежащий неизвестный предмет, нахождение клиента снаружи или неизвестное начало переноса не доказывают внос.",
              "наблюдается внос того же средства связи работником внутрь кассового узла"),
        _fact("WP_CONFIRMED_INDOOR_WORKPLACE", "Подтверждено ли помещение, являющееся рабочим местом или рабочей зоной?",
              "Нужно видимое закрытое помещение и положительно установленная рабочая функция зоны. Улица, клиентская зона неизвестного назначения, специально не классифицированное помещение и наличие одного стола не достаточны. Банковская принадлежность для этой нормы не требуется.",
              "подтверждено внутреннее помещение рабочего места или рабочей зоны"),
        _fact("WP_TOBACCO_PRODUCT_CLEAR", "Ясно ли различимо именно табачное изделие?",
              "Нужны достаточные визуальные признаки табачной сигареты/сигары или иного однозначно табачного изделия. Ручка, лекарственный ингалятор, еда, неизвестный вейп и неразличимый предмет не подходят. Не определять состав неизвестного пара и не выводить употребление алкоголя или наркотиков.",
              "ясно различимо табачное изделие"),
        _fact("WP_TOBACCO_CONSUMPTION_VISIBLE", "Наблюдается ли фактическое потребление того же табачного изделия?",
              "В исходной последовательности видны различимое табачное изделие и фактическое курение, например тлеющий конец и согласованная последовательность затяжки/дыма. Простое держание, движение руки ко рту, пар или конденсат сами по себе не достаточны.",
              "наблюдается фактическое курение различимого табачного изделия"),
    ]
    return {row["fact_id"]: row for row in rows}


def _health_source(root: Path):
    names = ["health_code_content.json", "health_code_versions.json",
             "health_code_article110.txt", "health_code_metadata.json"]
    archives = []
    expected_hashes = {
        "health_code_content.json": "e82209a5873bb32eec17a52d933e1bffd82195eee5300b6d3fe5d0bfd132c841",
        "health_code_versions.json": "558abfb39005f1123ecc580a6b836917cd7d2d503196593bfd263553afbfa27f",
        "health_code_article110.txt": "5229b29de075c16c7bbbae93121cf837d1168187a0f5a825b32345efd63f7af9",
        "health_code_metadata.json": "bfcb9880d243f6c1384904fea63246c29370ff9ba733b6ad2f94e1bc49008df6",
    }
    for name in names:
        path = root / "workplace_sources" / name
        data = path.read_bytes()
        if sha256(data).hexdigest() != expected_hashes[name]:
            raise ValueError("optional workplace official source archive changed")
        archives.append({"path": f"workplace_sources/{name}", "sha256": sha256(data).hexdigest(), "bytes": len(data)})
    return {
        "source_id": "RK_HEALTH_CODE_110_CURRENT_OPTIONAL", "url": "https://adilet.zan.kz/rus/docs/K2000000360",
        "document_title": "Кодекс Республики Казахстан «О здоровье народа и системе здравоохранения» от 7 июля 2020 года № 360-VI",
        "legal_status": "current_official_version", "accessed_on": "2026-10-17",
        "official_document_id": "44307b10-cd74-4f8e-86e5-5e5a10179e99",
        "primary_sha256": archives[0]["sha256"], "archives": archives,
        "text_archive_path": "workplace_sources/health_code_article110.txt",
        "version_boundary_utc": "2026-09-05T18:00:00Z",
        "provenance": "Unmodified official content/history responses; deterministic article extraction and explicitly derived metadata snapshot.",
    }


def build_workplace_payload():
    baseline = load_catalog()
    root = CATALOG_PATH.parent
    bank = deepcopy(next(iter(baseline.sources.values())))
    health = _health_source(root)
    facts = _new_facts()
    common = ["WP_CONFIRMED_BANK_CASH_NODE", "WP_CASH_NODE_BOUNDARY_VISIBLE",
              "WP_WORKER_ROLE_BY_CASH_ACTIONS", "WP_PERSONAL_ITEM_PROVENANCE_CLEAR",
              "WP_WORKER_ITEM_TRACK_CONTINUOUS", "WORK_AREA_CONTINUOUSLY_VISIBLE"]
    specs = [
        {"id": RULE_IDS[0], "event": "OPT_PERSONAL_STORAGE_IN_CASH_NODE", "label": "Возможное хранение личных вещей в кассовом узле",
         "source": bank, "clause": "пункт 12, часть первая, с учетом части второй",
         "excerpt": "12. Хранение личных вещей, наличных денег и ценностей работников банка осуществляется вне кассового узла в порядке, установленном внутренними документами банка.",
         "present": common + ["WP_PERSONAL_STORAGE_EPISODE_VISIBLE"], "absent": ["WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM"],
         "actions": ["WP_PERSONAL_STORAGE_EPISODE_VISIBLE"],
         "description": ["WP_PERSONAL_STORAGE_EPISODE_VISIBLE", "WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM"],
         "physical": "Прослеженный личный предмет работника оставлен на хранение в подтвержденной внутренней зоне кассового узла; предмет положительно отличим от лекарства, медицинского изделия и средства связи без камеры.",
         "exceptions": "Лекарства, медицинские изделия и средства связи без функции фото/видео; неизвестное содержимое, банковское оборудование и клиентское имущество исключаются положительными фактами.",
         "implication": "Возможные признаки хранения личных вещей работника внутри кассового узла при исключенных разрешенных категориях пункта 12."},
        {"id": RULE_IDS[1], "event": "OPT_RECORDING_COMM_DEVICE_IN_CASH_NODE", "label": "Возможный внос средства связи с фото- или видеофиксацией в кассовый узел",
         "source": bank, "clause": "пункт 12, часть вторая, во взаимосвязи с частью первой",
         "excerpt": "Допускается внос работниками банка в кассовый узел лекарственных средств, медицинских изделий и средств связи, не имеющих функции фото и видео фиксации.",
         "present": common + ["WP_COMM_DEVICE_IDENTIFIABLE", "WP_RECORDING_CAPABILITY_VISIBLE", "WP_DEVICE_ENTRY_INTO_NODE_VISIBLE"],
         "absent": ["WP_ALLOWED_MEDICAL_OR_NONCAMERA_COMM"], "actions": ["WP_DEVICE_ENTRY_INTO_NODE_VISIBLE"],
         "description": ["WP_DEVICE_ENTRY_INTO_NODE_VISIBLE", "WP_RECORDING_CAPABILITY_VISIBLE"],
         "physical": "Тот же работник переносит прослеженное личное средство связи с ясно установленной фото- или видеофункцией через подтвержденную границу внутрь кассового узла.",
         "exceptions": "Средства связи без фото/видео, медицинские изделия, неизвестные прямоугольные предметы, клиентские устройства снаружи узла и штатное банковское оборудование не удовлетворяют узкой ветви.",
         "implication": "Возможные признаки вноса личного средства связи с наблюдаемо подтвержденной функцией фото- или видеофиксации внутрь кассового узла; разрешение части второй пункта 12 относится к средствам связи без такой функции."},
        {"id": RULE_IDS[2], "event": "OPT_TOBACCO_CONSUMPTION_INDOOR_WORKPLACE", "label": "Возможное курение табачного изделия в помещении рабочей зоны",
         "source": health, "clause": "подпункт 9 пункта 5 статьи 110 с учетом пункта 6",
         "excerpt": "9) в помещениях, являющихся рабочими местами и рабочими зонами;",
         "present": ["WP_CONFIRMED_INDOOR_WORKPLACE", "WP_TOBACCO_PRODUCT_CLEAR", "WP_TOBACCO_CONSUMPTION_VISIBLE", "WORK_AREA_CONTINUOUSLY_VISIBLE"],
         "absent": [], "actions": ["WP_TOBACCO_CONSUMPTION_VISIBLE"],
         "description": ["WP_CONFIRMED_INDOOR_WORKPLACE", "WP_TOBACCO_CONSUMPTION_VISIBLE"],
         "physical": "Внутри подтвержденной рабочей зоны наблюдается фактическое курение ясно различимого табачного изделия; неизвестный пар и простое держание предмета недостаточны.",
         "exceptions": "Неизвестный предмет/вейп, лекарственный ингалятор, простое держание изделия и улица не подходят. Исключение пункта 6 относится к подпунктам 3 и 7 пункта 5, а не к рабочей зоне по подпункту 9.",
         "implication": "Возможные признаки потребления ясно различимого табачного изделия в помещении рабочего места или рабочей зоны, где действует запрет подпункта 9 пункта 5 статьи 110."},
    ]
    records, bindings, events = [], {}, {}
    for spec in specs:
        states = dict.fromkeys(spec["present"], "present")
        states.update(dict.fromkeys(spec["absent"], "absent"))
        source = spec["source"]
        records.append({"rule_id": spec["id"], "title": source["document_title"], "clause": spec["clause"],
            "source_url": source["url"], "source_publisher": "Официальная ИПС «Әділет»", "accessed_at": "2026-10-17",
            "source_sha256": source["primary_sha256"], "jurisdiction": "KZ",
            "applicability": "Изолированная опциональная ветвь; каждый положительный факт и исключение должны подтверждаться исходными кадрами и отдельным доверенным контекстом зоны.",
            "required_facts": list(spec["present"]), "temporal_conditions": ["Непрерывная связь обязательных наблюдений относится к тому же работнику/предмету; действие подтверждено внутри интервала кандидата."],
            "exceptions": [spec["exceptions"]], "verification_state": "source_verified", "applicability_verified": True,
            "enabled": True, "notes": "Enabled only inside explicit optional loader; production profile selection is absent and unmeasured."})
        bindings[spec["id"]] = {"source_id": source["source_id"], "clause": spec["clause"], "source_excerpt": spec["excerpt"],
            "event_type_id": spec["event"], "required_fact_states": states,
            "applicability_fact_ids": [key for key in spec["present"] if key not in spec["actions"]],
            "violation_fact_ids": spec["actions"], "context_fact_ids": ["WORK_AREA_CONTINUOUSLY_VISIBLE"],
            "required_context_fact_ids": ["WORK_AREA_CONTINUOUSLY_VISIBLE"],
            "exception_fact_ids": list(spec["absent"]), "temporal_verified": True, "exceptions_verified": True,
            "is_omission": False, "description_fact_ids": spec["description"], "implication_ru": spec["implication"],
            "profile_scope": "optional_workplace_only", "anchor_policy": "first_supported_action",
            "remaining_limit": "Composite continuity/scope predicates require actual local visual proof; the builder does not independently recover tracks or identify devices."}
        events[spec["event"]] = {"name_ru": spec["label"], "required_fact_states": states,
                                "definition_ru": spec["physical"]}
    axes = [fact.fact_id for fact in baseline.fact_definitions()] + sorted(facts)
    return {"schema_version": SCHEMA, "catalog_version": VERSION, "accessed_on": "2026-10-17",
        "deployment_selected": False, "independent_accuracy_measured": False,
        "baseline_snapshot": {"catalog_sha256": sha256(CATALOG_PATH.read_bytes()).hexdigest(),
                              "taxonomy_sha256": sha256((root.parent / "annotations/taxonomy.v1.json").read_bytes()).hexdigest()},
        "vocabulary": {"version": VOCABULARY_VERSION, "fact_ids": axes,
                       "fact_axis_sha256": digest(axes), "state_order": ["absent", "present", "unclear"],
                       "baseline_prefix_length": 19, "new_axis_ids": sorted(facts)},
        "sources": [bank, health], "rules": records, "bindings": bindings,
        "facts": {**deepcopy(baseline.facts), **facts}, "event_types": events, "coverage_gaps": [],
        "approved_limitations_ru": ["Результат является возможным признаком и требует проверки специалистом."]}


class WorkplaceRuleCatalog(RuleCatalog):
    def __init__(self, payload: dict[str, Any], *, root: Path, verify_archives: bool = True,
                 include_baseline: bool = False):
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("unsupported optional workplace schema")
        baseline = load_catalog()
        if payload["baseline_snapshot"]["catalog_sha256"] != sha256(CATALOG_PATH.read_bytes()).hexdigest():
            raise ValueError("optional workplace baseline pin mismatch")
        if payload["baseline_snapshot"]["taxonomy_sha256"] != sha256(
            (CATALOG_PATH.parent.parent / "annotations/taxonomy.v1.json").read_bytes()
        ).hexdigest():
            raise ValueError("optional workplace taxonomy pin mismatch")
        axis = payload["vocabulary"]
        prefix = [fact.fact_id for fact in baseline.fact_definitions()]
        if (axis["version"] != VOCABULARY_VERSION or axis["fact_ids"][:19] != prefix
                or len(axis["fact_ids"]) != 32 or axis["fact_axis_sha256"] != digest(axis["fact_ids"])
                or axis["state_order"] != ["absent", "present", "unclear"]):
            raise ValueError("optional workplace axis identity mismatch")
        if set(record["rule_id"] for record in payload["rules"]) != set(RULE_IDS):
            raise ValueError("optional workplace rule set mismatch")
        expected = build_workplace_payload()
        for key in ("sources", "rules", "bindings", "facts", "event_types", "approved_limitations_ru"):
            if payload[key] != expected[key]:
                raise ValueError(f"optional workplace reviewed {key} differ")
        if payload.get("deployment_selected") is not False or payload.get("independent_accuracy_measured") is not False:
            raise ValueError("optional workplace profile is not selected or measured")
        self.profile_payload = deepcopy(payload)
        self.vocabulary = deepcopy(axis)
        self.vocabulary_version = axis["version"]
        adapted = deepcopy(payload)
        adapted["schema_version"] = "forte.rules.v1"
        if include_baseline:
            adapted["rules"] = deepcopy(baseline.payload["rules"]) + adapted["rules"]
            adapted["bindings"] = {**deepcopy(baseline.bindings), **adapted["bindings"]}
            adapted["event_types"] = {**deepcopy(baseline.event_types), **adapted["event_types"]}
            adapted["coverage_gaps"] = deepcopy(baseline.coverage_gaps)
        super().__init__(adapted, root=root, verify_archives=verify_archives)
        self.is_optional = True
        self.deployment_selected = False
        self.includes_baseline = include_baseline

    def fact_definitions(self, *, enabled_only=True, trusted_scene_context=None):
        base_context = "bank_cash_workstation" if trusted_scene_context == "bank_cash_node" else trusted_scene_context
        values = super().fact_definitions(enabled_only=enabled_only, trusted_scene_context=base_context)
        result = []
        for value in values:
            if value.fact_id == "WP_CONFIRMED_BANK_CASH_NODE":
                suffix = (" Отдельная доверенная настройка bank_cash_node задана; дополнительно подтвердите по кадрам внутреннюю зону."
                          if trusted_scene_context == "bank_cash_node" else
                          " Отдельная доверенная настройка bank_cash_node не задана: состояние должно быть unclear.")
                value = value.model_copy(update={"description": value.description + suffix})
            result.append(value)
        return result

    def axis_fact_definitions(self, *, trusted_scene_context=None):
        values = {value.fact_id: value for value in self.fact_definitions(enabled_only=False, trusted_scene_context=trusted_scene_context)}
        return [values[key] for key in self.vocabulary["fact_ids"]]

    def validate_axis_order(self, fact_ids, *, vocabulary_version):
        if vocabulary_version != self.vocabulary_version:
            raise ValueError("optional workplace vocabulary version mismatch")
        if list(fact_ids) != self.vocabulary["fact_ids"]:
            raise ValueError("optional workplace axis order mismatch")


def load_workplace_catalog(path: Path | None = None, *, verify_archives=True, include_baseline=False):
    path = Path(path) if path is not None else PROFILE_PATH
    return WorkplaceRuleCatalog(json.loads(path.read_text(encoding="utf-8")), root=path.parent,
                                verify_archives=verify_archives, include_baseline=include_baseline)


def build_workplace_taxonomy(payload):
    return {"schema_version": "forte.optional.workplace.taxonomy.v1", "language": "ru",
            "status": "optional_unselected_unmeasured", "catalog_version": VERSION,
            "vocabulary": payload["vocabulary"],
            "facts": [payload["facts"][key] for key in payload["vocabulary"]["fact_ids"]],
            "event_types": [{"event_type_id": key, **value} for key, value in payload["event_types"].items()]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = build_workplace_payload()
    if args.write:
        PROFILE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        TAXONOMY_PATH.write_text(json.dumps(build_workplace_taxonomy(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps({"schema_version": SCHEMA, "rules": len(payload["rules"]),
                          "axes": len(payload["vocabulary"]["fact_ids"]), "deployment_selected": False}))

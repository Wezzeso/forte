"""Explicitly opt-in rule vocabulary; never modifies the default legal catalog.

The raw JSON deliberately has a different schema so the production loader
rejects it. Only ``load_experimental_catalog`` opts into these four additional
branches. Callers must use A11's experimental temporal proof wrapper, never the
ordinary co-occurrence aggregator for these rules.
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

EXPERIMENTAL_CATALOG_PATH = Path(__file__).with_name("experimental_catalog.json")
EXPERIMENTAL_SCHEMA = "forte.experimental.rules.v1"
EXPERIMENTAL_VERSION = "2026-10-17.priority1.observable.v1"
VOCABULARY_VERSION = "forte.experimental.fact_axes.priority1.v1"
TEMPORAL_SCHEMA = "forte.experimental.temporal.v1"

RULE_IDS = (
    "EXP_RK_NBK231_P20_PRIOR_CASH_OUTSIDE_STORAGE",
    "EXP_RK_NBK231_P20_RECEIPT_CLIENT_ABSENT",
    "EXP_RK_NBK231_P19_2_COIN_RECEIPT_COUNT",
    "EXP_RK_NBK231_P32_DAMAGED_BUNDLE_COUNT",
)


def _digest(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def _fact(fact_id: str, question: str, definition: str, phrase: str,
          absence_phrase: str | None = None) -> dict[str, str]:
    result = {"fact_id": fact_id, "question_ru": question, "definition_ru": definition, "description_ru": phrase}
    if absence_phrase:
        result["absence_description_ru"] = absence_phrase
    return result


def _additional_facts() -> dict[str, dict[str, str]]:
    rows = [
        _fact("EXP_PRIOR_RECEIPT_START_VISIBLE", "Видно ли начало именно предыдущего приема отслеживаемой наличности?",
              "Граница начала приема подтверждена исходными кадрами передачи наличных от клиента. Запись, начавшаяся после приема, не подтверждает этот факт. Это граница отдельного эпизода процедуры, а не установление личности клиента.",
              "видно начало предыдущего приема отслеживаемой наличности"),
        _fact("EXP_NEXT_RECEIPT_START_VISIBLE", "Видно ли начало следующего отдельного приема после завершения предыдущего?",
              "Новая передача наличных начинается после видимого завершения предыдущего приема. Предыдущий и текущий эпизоды должны иметь разные локальные procedure_id. Не считать новым приемом продолжение, паузу или вторую руку в том же незавершенном действии.",
              "после предыдущего завершенного приема начинается следующий отдельный прием"),
        _fact("EXP_NEXT_RECEIPT_ACTIVE", "Идет ли следующий отдельный прием в момент нахождения ранее принятых денег на поверхности?",
              "На исходном кадре подтвержден активный этап нового приема от клиента и одновременно прослеженные ранее принятые деньги. Две отдельные операции устанавливаются по последовательности, без распознавания лиц.",
              "следующий отдельный прием наличных уже идет"),
        _fact("EXP_RECEIPT_EPISODES_DISTINCT", "Подтверждены ли два различных завершенный/последующий эпизода приема?",
              "Полный контекст показывает завершенный предыдущий прием, затем начало нового приема. Локальные идентификаторы эпизодов различаются и связаны с исходными кадрами. Различие имен, лиц или предположенная принадлежность денег не используются.",
              "предыдущий и последующий приемы являются разными эпизодами процедуры"),
        _fact("EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER", "Остаются ли прослеженные ранее принятые деньги без обработки на обычной поверхности во время следующего приема?",
              "Одна и та же ранее принятая наличность лежит на обычной рабочей поверхности вне металлического шкафа, сейфа, тележки, ящика стола или тумбы, предназначенных для хранения и закрывающихся на ключ. В этот момент эти деньги не пересчитываются, не упаковываются и не передаются на следующий внутренний этап. Нельзя считать нарушением само открытие устройства или нахождение денег внутри предназначенного для хранения ящика.",
              "ранее принятая наличность без обработки остается на обычной поверхности вне устройств хранения"),
        _fact("EXP_CASH_TRACK_CONTINUOUS", "Прослежена ли одна и та же наличность через все обязательные этапы без подмены или неизвестного разрыва?",
              "Один локальный cash_track_id связан с исходными кадрами всех требуемых фаз. Виден непрерывный путь объекта; смешение двух стопок, заслонение с возможной подменой и выход из наблюдаемой зоны требуют unclear. Факт не определяет владельца, сумму, номинал, умысел или разрешение.",
              "одна и та же наличность непрерывно прослежена между обязательными этапами"),
        _fact("EXP_RECEIPT_START_VISIBLE", "Видно ли начало конкретного приема наличных от клиента кассовым работником?",
              "Исходные кадры показывают начало передачи наличных от клиента в конкретном эпизоде приема. Если подготовка/прием начались раньше записи или вне кадра, состояние unclear.",
              "видно начало конкретного приема наличных от клиента"),
        _fact("EXP_RECEIPT_COMPLETE_VISIBLE", "Видна ли граница завершения этого же приема наличных?",
              "Окончание релевантного приема подтверждено исходным контекстом и относится к тому же procedure_id и cash_track_id. Уход человека, пауза, окончание ролика или выход предмета из кадра сами по себе не доказывают завершение.",
              "видно завершение того же приема наличных"),
        _fact("EXP_CONTRIBUTOR_BODY_TRACK_CONTINUOUS", "Прослежено ли тело внесшего наличные человека от передачи до выхода из клиентской зоны?",
              "Непрерывность анонимного локального contributor_track_id устанавливается по движению тела в исходных кадрах передачи и выхода. Не распознавать или сопоставлять лица, имена и личность. Трек не требуется в фазе, где человек уже физически отсутствует.",
              "анонимный телесный трек вносившего наличные прослежен от передачи до выхода"),
        _fact("EXP_CONTRIBUTOR_LEAVES_CLIENT_AREA", "Виден ли фактический выход вносившего наличные человека из всей рассматриваемой клиентской зоны?",
              "Виден выход за надежно определенную границу зоны присутствия клиента после передачи наличных и до пересчета того же незавершенного приема. Наклон, заслонение, перемещение в пределах зоны или изменение кропа выходом не считаются.",
              "вносивший наличные человек видимо выходит из клиентской зоны до пересчета"),
        _fact("EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE", "Полностью и непрерывно видны ли все места возможного присутствия клиента при этом приеме?",
              "Виден весь относящийся к кассе клиентский участок и возможные заслоненные/соседние положения, от передачи до пересчета. Если клиент или его представитель может находиться в слепой зоне, вне кропа или за перегородкой, состояние unclear. Не выводить отсутствие по одному пустому фрагменту поверхности.",
              "все места возможного присутствия клиента непрерывно доступны наблюдению"),
        _fact("EXP_ANY_CLIENT_AREA_PERSON_PRESENT", "Находится ли хоть один человек в относящейся к кассе клиентской зоне во время приема-пересчета?",
              "Оценивается физическое присутствие любого человека на стороне клиента в фазе пересчета, без определения личности, полномочий или представительства. absent разрешено только при полном непрерывном обзоре всей зоны и подтвержденной ее пустоте на всем пересчете. Любой находящийся там человек исключает эту узкую ветвь независимо от предполагаемой роли.",
              "в клиентской зоне физически присутствует человек",
              "на всем наблюдаемом этапе пересчета клиентская зона физически пуста"),
        _fact("EXP_COUNT_DURING_ACTIVE_RECEIPT", "Пересчитываются ли те же вносимые деньги до завершения именно клиентского приема?",
              "Наблюдается ручной или машинный пересчет той же наличности после клиентской передачи и до завершения того же приема. Самостоятельный внутренний повторный пересчет после завершенного обслуживания не подходит.",
              "те же деньги пересчитываются в еще не завершенном клиентском приеме"),
        _fact("EXP_LOOSE_COINS_RECEIVED", "Принимаются ли от клиента различимые отдельные неупакованные монеты?",
              "Видна передача отдельных монет в конкретном клиентском приеме. Не применять к банкнотам, жетонам, бумаге, закрытой сумке инкассации или неизвестному содержимому пакета. Монетный номинал и сумма не требуются.",
              "от клиента принимаются отдельные неупакованные монеты"),
        _fact("EXP_ISSUE_PREPARATION_START_VISIBLE", "Видно ли начало полного этапа подготовки рассматриваемой наличности к выдаче?",
              "Исходная последовательность охватывает подготовку до любого возможного ручного или машинного пересчета. Если деньги могли быть пересчитаны до начала записи, состояние unclear. Та же наличность прослеживается до передачи клиенту.",
              "подготовка той же наличности к выдаче видна с начала"),
        _fact("EXP_ISSUE_HANDOVER_COMPLETE_VISIBLE", "Виден ли завершенный этап передачи той же наличности клиенту?",
              "Кадры показывают границу завершения конкретной выдачи и относятся к тому же procedure_id и cash_track_id. Не заменять фактическое завершение окончанием ролика, заслонением или паузой.",
              "видно завершение передачи той же наличности клиенту"),
        _fact("EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "Можно ли уверенно различить именно корешок банкнот, а не внешнюю сумку, пачку или отдельный документ?",
              "Достаточное разрешение и вид упаковки позволяют установить единицу «корешок банкнот» для пункта 32. Число листов, сумму или номинал по приблизительному размеру не предполагать. Если видна только внешняя сумка/конверт или единица упаковки неоднозначна, unclear.",
              "различим именно корешок банкнот"),
        _fact("EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR", "Явно ли повреждена упаковка самого выдаваемого корешка до передачи клиенту?",
              "На исходном детальном кадре до выдачи виден разрыв, вскрытие или нарушение упаковки самого корешка банкнот. Тень, складка, дефект внешней сумки, снятый внешний пакет при неповрежденном корешке и недостаточное разрешение не подтверждают факт. Неповрежденные пачки/корешки исключаются.",
              "до передачи явно повреждена упаковка самого корешка банкнот"),
    ]
    return {row["fact_id"]: row for row in rows}


def _temporal(phases: list[tuple[str, list[str]]], *, boundary_facts: list[str],
              track_phases: dict[str, list[str]], track_facts: dict[str, str],
              distinct: bool = False, roles: dict[str, str] | None = None,
              absence: bool = False, counting_paths: bool = False) -> dict[str, Any]:
    return {
        "schema_version": TEMPORAL_SCHEMA,
        "ordered_phases": [{"phase_id": key, "fact_ids": values} for key, values in phases],
        "distinct_procedure_ids": distinct,
        "phase_procedure_roles": roles or {key: "current" for key, _ in phases},
        "required_track_kinds": sorted(track_phases),
        "track_phase_ids": track_phases,
        "track_fact_ids": track_facts,
        "absence_must_cover_procedure": absence,
        "absence_phase_ids": [],
        "counting_paths_must_cover_procedure": counting_paths,
        "required_procedure_boundary_fact_ids": boundary_facts,
        "strict_phase_order": True,
        "max_visibility_gap_sec": None,
        "visibility_gap_policy": "no_unknown_gap_or_blind_area_in_required_procedure",
        "context_is_automated_local_evidence": True,
        "face_identification_permitted": False,
    }


def build_experimental_payload() -> dict[str, Any]:
    """Build a reproducible JSON payload without writing or enabling defaults."""
    baseline = load_catalog()
    source = deepcopy(baseline.payload["sources"][0])
    facts = deepcopy(baseline.facts)
    facts.update(_additional_facts())
    previous_excerpt = baseline.bindings["RK_NBK231_P20_PREVIOUS_CASH_POCKET"]["source_excerpt"]
    coin_excerpt = baseline.bindings["RK_NBK231_P19_2_RECEIPT_COUNT"]["source_excerpt"]
    text = (baseline.root / source["text_archive_path"]).read_text(encoding="utf-8")
    damaged_excerpt = "Банкноты из неполных, а также поврежденных корешков выдаются кассовым работником полистным пересчетом."
    assert damaged_excerpt in text
    common = ["BANK_CASH_WORKSTATION", "CASH_VISIBLE", "WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "EXP_CASH_TRACK_CONTINUOUS"]
    definitions = [
        {
            "rule_id": RULE_IDS[0], "event_type_id": "EXP_PREVIOUS_CASH_OUTSIDE_STORAGE", "clause": "пункт 20, часть вторая",
            "source_excerpt": previous_excerpt,
            "present": common + ["CASH_RECEIPT_OPERATION", "PREVIOUS_RECEIPT_COMPLETE", "PREVIOUSLY_ACCEPTED_CASH_TRACKED", "EXP_RECEIPT_EPISODES_DISTINCT", "EXP_NEXT_RECEIPT_ACTIVE", "EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER"],
            "absent": [],
            "applicability": "Прослеженные ранее принятые деньги после завершенного предыдущего приема остаются без обработки на обычной поверхности вне устройств хранения во время следующего отдельного приема.",
            "exceptions": ["Исключаются наличные текущего незавершенного приема, предназначенные для хранения ящики/кассеты, текущий внутренний пересчет, упаковка или передача этой наличности."],
            "description_fact_ids": ["PREVIOUSLY_ACCEPTED_CASH_TRACKED", "EXP_NEXT_RECEIPT_ACTIVE", "EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER"],
            "implication_ru": "Возможные признаки хранения ранее принятых наличных вне предусмотренных пунктом 20 устройств во время следующего приема.",
            "temporal": _temporal([
                ("prior_receipt_start", ["EXP_PRIOR_RECEIPT_START_VISIBLE"]),
                ("prior_receipt_complete", ["PREVIOUS_RECEIPT_COMPLETE"]),
                ("next_receipt_start", ["EXP_NEXT_RECEIPT_START_VISIBLE"]),
                ("prior_cash_idle_during_next_receipt", ["EXP_NEXT_RECEIPT_ACTIVE", "EXP_PRIOR_CASH_IDLE_ON_ORDINARY_COUNTER"]),
            ], boundary_facts=["EXP_PRIOR_RECEIPT_START_VISIBLE", "PREVIOUS_RECEIPT_COMPLETE", "EXP_NEXT_RECEIPT_START_VISIBLE"],
                track_phases={"cash": ["prior_receipt_start", "prior_receipt_complete", "next_receipt_start", "prior_cash_idle_during_next_receipt"]},
                track_facts={"cash": "EXP_CASH_TRACK_CONTINUOUS"}, distinct=True,
                roles={"prior_receipt_start": "prior", "prior_receipt_complete": "prior", "next_receipt_start": "current", "prior_cash_idle_during_next_receipt": "current"}),
            "omission": False,
        },
        {
            "rule_id": RULE_IDS[1], "event_type_id": "EXP_CLIENT_ABSENT_DURING_RECEIPT_COUNT", "clause": "пункт 20, часть первая",
            "source_excerpt": "20. Наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента.",
            "present": common + ["CASH_RECEIPT_OPERATION", "CASH_COUNT_VISIBLE", "EXP_COUNT_DURING_ACTIVE_RECEIPT", "EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE", "EXP_CONTRIBUTOR_BODY_TRACK_CONTINUOUS", "EXP_CONTRIBUTOR_LEAVES_CLIENT_AREA"],
            "absent": ["EXP_ANY_CLIENT_AREA_PERSON_PRESENT"],
            "applicability": "Прием и пересчет той же наличности до завершения клиентского приема; после видимого ухода вносившего деньги человека вся клиентская зона физически пуста на протяжении пересчета.",
            "exceptions": ["Исключаются самостоятельный внутренний повторный пересчет после обслуживания, неполный обзор клиентской зоны, скрытый клиент/представитель и любой человек в клиентской зоне независимо от предполагаемой роли."],
            "description_fact_ids": ["EXP_COUNT_DURING_ACTIVE_RECEIPT", "EXP_ANY_CLIENT_AREA_PERSON_PRESENT", "EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE"],
            "implication_ru": "Возможные признаки пересчета в рамках приема наличных без физического присутствия клиента, предусмотренного пунктом 20.",
            "temporal": _temporal([
                ("contribution", ["EXP_RECEIPT_START_VISIBLE", "CASH_HANDOVER_VISIBLE"]),
                ("client_leaves", ["EXP_CONTRIBUTOR_LEAVES_CLIENT_AREA"]),
                ("count_with_client_area_empty", ["EXP_COUNT_DURING_ACTIVE_RECEIPT", "CASH_COUNT_VISIBLE", "EXP_ANY_CLIENT_AREA_PERSON_PRESENT"]),
                ("receipt_complete", ["EXP_RECEIPT_COMPLETE_VISIBLE"]),
            ], boundary_facts=["EXP_RECEIPT_START_VISIBLE", "EXP_RECEIPT_COMPLETE_VISIBLE"],
                track_phases={"cash": ["contribution", "count_with_client_area_empty", "receipt_complete"], "contributor": ["contribution", "client_leaves"]},
                track_facts={"cash": "EXP_CASH_TRACK_CONTINUOUS", "contributor": "EXP_CONTRIBUTOR_BODY_TRACK_CONTINUOUS"}),
            # This branch observes counting while the entire client side is
            # empty. It does not infer a missing count or another missing act.
            "omission": False,
        },
        {
            "rule_id": RULE_IDS[2], "event_type_id": "CASH_TRANSFER_WITHOUT_OBSERVED_COUNT", "clause": "подпункт 2 пункта 19 (монеты)",
            "source_excerpt": coin_excerpt,
            "present": common + ["CASH_RECEIPT_OPERATION", "EXP_LOOSE_COINS_RECEIVED", "CASH_HANDOVER_VISIBLE", "ALL_COUNTING_PATHS_VISIBLE"],
            "absent": ["CASH_COUNT_VISIBLE"],
            "applicability": "Завершенный полностью наблюдаемый прием отдельных монет от клиента; отсутствует ручной или машинный пересчет по кружкам на всем этапе конкретного приема.",
            "exceptions": ["Исключаются закрытые сумки инкассации и неизвестное содержимое, этап до записи, скрытый счетчик монет и автоматический путь приема вне наблюдения. Пункт 6 допускает автоматические устройства."],
            "description_fact_ids": ["EXP_LOOSE_COINS_RECEIVED", "CASH_COUNT_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE"],
            "implication_ru": "Возможные признаки приема отдельных монет без наблюдаемого обязательного пересчета по кружкам.",
            "temporal": _temporal([
                ("procedure_start", ["EXP_RECEIPT_START_VISIBLE", "EXP_LOOSE_COINS_RECEIVED"]),
                ("no_count_handover", ["CASH_HANDOVER_VISIBLE"]),
                ("procedure_complete", ["EXP_RECEIPT_COMPLETE_VISIBLE"]),
            ], boundary_facts=["EXP_RECEIPT_START_VISIBLE", "EXP_RECEIPT_COMPLETE_VISIBLE"],
                track_phases={"cash": ["procedure_start", "no_count_handover", "procedure_complete"]}, track_facts={"cash": "EXP_CASH_TRACK_CONTINUOUS"},
                absence=True, counting_paths=True),
            "omission": True,
        },
        {
            "rule_id": RULE_IDS[3], "event_type_id": "CASH_TRANSFER_WITHOUT_OBSERVED_COUNT", "clause": "пункт 32, часть первая, второе предложение, с учетом подпункта 4 пункта 31",
            "source_excerpt": damaged_excerpt,
            "present": common + ["CASH_ISSUE_OPERATION", "EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR", "CASH_HANDOVER_VISIBLE", "ALL_COUNTING_PATHS_VISIBLE"],
            "absent": ["CASH_COUNT_VISIBLE"],
            "applicability": "Именно поврежденный корешок банкнот, а не внешняя сумка, выдается клиенту; полный этап подготовки и выдачи прослежен без ручного или машинного полистного пересчета.",
            "exceptions": ["Исключаются целые неповрежденные пачки и корешки, выдаваемые по сумме на упаковке согласно пункту 32; повреждение только внешней сумки; неизвестная единица упаковки; пересчет до начала видимого этапа или скрытым устройством."],
            "description_fact_ids": ["EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR", "CASH_COUNT_VISIBLE"],
            "implication_ru": "Возможные признаки выдачи поврежденного корешка банкнот без наблюдаемого обязательного полистного пересчета.",
            "temporal": _temporal([
                ("procedure_start", ["EXP_ISSUE_PREPARATION_START_VISIBLE", "EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR"]),
                ("no_count_handover", ["CASH_HANDOVER_VISIBLE"]),
                ("procedure_complete", ["EXP_ISSUE_HANDOVER_COMPLETE_VISIBLE"]),
            ], boundary_facts=["EXP_ISSUE_PREPARATION_START_VISIBLE", "EXP_ISSUE_HANDOVER_COMPLETE_VISIBLE"],
                track_phases={"cash": ["procedure_start", "no_count_handover", "procedure_complete"]}, track_facts={"cash": "EXP_CASH_TRACK_CONTINUOUS"},
                absence=True, counting_paths=True),
            "omission": True,
        },
    ]
    definitions[1]["temporal"]["absence_phase_ids"] = ["count_with_client_area_empty"]
    for item, event_phase in zip(definitions, ["prior_cash_idle_during_next_receipt", "count_with_client_area_empty", "procedure_complete", "procedure_complete"], strict=True):
        item["temporal"]["event_phase_id"] = event_phase
        item["temporal"]["candidate_interval_policy"] = (
            "full_procedure_source_bounds" if item["omission"] else "event_phase_source_bounds"
        )
    records = []
    bindings = {}
    for item in definitions:
        states = {key: "present" for key in dict.fromkeys(item["present"])}
        states.update({key: "absent" for key in item["absent"]})
        temporal = item["temporal"]
        description = ["Фазы должны быть подтверждены исходными кадрами строго в указанном порядке: "
                       + " → ".join(p["phase_id"] for p in temporal["ordered_phases"]) + ".",
                       "Треки наличности/тела и идентификаторы процедур привязаны к исходным FrameRef; пропуск или неизвестный участок не подтверждает отсутствие действия."]
        if item["omission"]:
            description.append("Отсутствие пересчета и видимость всех ручных/машинных путей должны подтверждать весь этап от начала подготовки/приема до завершения передачи/приема.")
        records.append({
            "rule_id": item["rule_id"], "title": source["document_title"], "clause": item["clause"],
            "source_url": source["url"], "source_publisher": "Правление Национального Банка Республики Казахстан; ИПС «Әділет»",
            "accessed_at": source["accessed_on"], "source_sha256": source["primary_sha256"], "jurisdiction": "KZ",
            "applicability": item["applicability"], "required_facts": [key for key, value in states.items() if value == "present"],
            "temporal_conditions": description, "exceptions": item["exceptions"], "verification_state": "verified",
            "applicability_verified": True, "enabled": True,
            "notes": "Экспериментальная необязательная ветвь. Только явный experimental loader и детерминированное доказательство фаз A11. Это не результат по частному видео и не активация производственного каталога.",
        })
        bindings[item["rule_id"]] = {
            "source_id": source["source_id"], "clause": item["clause"], "source_excerpt": item["source_excerpt"],
            "event_type_id": item["event_type_id"], "required_fact_states": states,
            "applicability_fact_ids": [key for key in states if key in {"BANK_CASH_WORKSTATION", "CASH_RECEIPT_OPERATION", "CASH_ISSUE_OPERATION", "PREVIOUS_RECEIPT_COMPLETE", "PREVIOUSLY_ACCEPTED_CASH_TRACKED", "EXP_RECEIPT_EPISODES_DISTINCT", "EXP_LOOSE_COINS_RECEIVED", "EXP_BANKNOTE_BUNDLE_IDENTIFIABLE", "EXP_BUNDLE_WRAPPER_DAMAGE_CLEAR"}],
            "context_fact_ids": [key for key in states if key in {"WORK_AREA_CONTINUOUSLY_VISIBLE", "PROCEDURE_CONTEXT_COMPLETE", "ALL_COUNTING_PATHS_VISIBLE", "EXP_ALL_CLIENT_AREAS_CONTINUOUSLY_VISIBLE", "EXP_CASH_TRACK_CONTINUOUS", "EXP_CONTRIBUTOR_BODY_TRACK_CONTINUOUS"}],
            "is_omission": item["omission"], "omission": item["omission"],
            "context_complete_fact_id": "PROCEDURE_CONTEXT_COMPLETE", "visibility_fact_id": "WORK_AREA_CONTINUOUSLY_VISIBLE",
            "temporal_verified": True, "exceptions_verified": True, "exception_fact_ids": [], "min_context_sec": None,
            "temporal_requirements": description, "experimental_temporal": temporal,
            "description_fact_ids": item["description_fact_ids"], "implication_ru": item["implication_ru"],
            "anchor_policy": "last_supported" if item["omission"] else "first_supported",
            "anchor_convention": "Последний исходный кадр на границе завершенного этапа процедуры." if item["omission"] else "Первый исходный кадр наблюдаемого действия в подтвержденной фазе события.",
            "experimental_only": True,
        }
    events = {
        "EXP_PREVIOUS_CASH_OUTSIDE_STORAGE": {"name_ru": "Возможное хранение ранее принятой наличности вне устройств хранения", "definition_ru": "Ранее принятая наличность прослежена на обычной поверхности во время следующего отдельного приема."},
        "EXP_CLIENT_ABSENT_DURING_RECEIPT_COUNT": {"name_ru": "Возможный пересчет при приеме без присутствия клиента", "definition_ru": "Пересчет того же незавершенного приема идет после ухода вносившего наличные человека при полностью пустой видимой клиентской зоне."},
        "CASH_TRANSFER_WITHOUT_OBSERVED_COUNT": deepcopy(baseline.event_types["CASH_TRANSFER_WITHOUT_OBSERVED_COUNT"]),
    }
    # Freeze the existing axis order separately. A 19-axis checkpoint is never
    # silently relabeled or expanded to satisfy this experimental vocabulary.
    baseline_axes = [value.fact_id for value in baseline.fact_definitions()]
    used = {key for binding in bindings.values() for key in binding["required_fact_states"]}
    for binding in bindings.values():
        for phase in binding["experimental_temporal"]["ordered_phases"]:
            used.update(phase["fact_ids"])
    axes = baseline_axes + sorted(used - set(baseline_axes))
    return {
        "schema_version": EXPERIMENTAL_SCHEMA, "catalog_version": EXPERIMENTAL_VERSION,
        "status": "optional_experimental_not_default", "accessed_on": source["accessed_on"],
        "requires_temporal_wrapper": True, "manual_annotation_prerequisite": False,
        "baseline_snapshot": {"catalog_sha256": sha256(CATALOG_PATH.read_bytes()).hexdigest(), "catalog_version": baseline.version,
                              "fact_axis_count": len(baseline_axes), "fact_axis_order": baseline_axes, "fact_axis_sha256": _digest(baseline_axes)},
        "vocabulary": {"schema_version": VOCABULARY_VERSION, "feature_axis_order": axes, "feature_axis_count": len(axes),
                       "feature_axis_sha256": _digest(axes), "baseline_axis_count": len(baseline_axes),
                       "new_axis_ids": axes[len(baseline_axes):], "model_axis_manifest_must_match": True,
                       "baseline_checkpoint_compatible": False},
        "experimental_rule_ids": list(RULE_IDS), "sources": [source], "rules": records, "bindings": bindings,
        "facts": facts, "event_types": events, "coverage_gaps": deepcopy(baseline.coverage_gaps),
        "approved_limitations_ru": deepcopy(baseline.payload["approved_limitations_ru"]),
    }


class ExperimentalRuleCatalog(RuleCatalog):
    """Explicit experimental metadata plus the existing source checks."""

    def __init__(self, payload: dict[str, Any], *, root: Path, include_baseline: bool = False):
        if payload.get("schema_version") != EXPERIMENTAL_SCHEMA or payload.get("requires_temporal_wrapper") is not True:
            raise ValueError("not an isolated experimental regulatory catalog")
        self.experimental_payload = deepcopy(payload)
        baseline = load_catalog()
        snapshot = payload["baseline_snapshot"]
        if sha256(CATALOG_PATH.read_bytes()).hexdigest() != snapshot["catalog_sha256"]:
            raise ValueError("baseline catalog changed; regenerate and review experimental snapshot")
        baseline_axes = [value.fact_id for value in baseline.fact_definitions()]
        if snapshot["fact_axis_order"] != baseline_axes or snapshot["fact_axis_sha256"] != _digest(baseline_axes):
            raise ValueError("baseline nineteen-axis vocabulary changed")
        vocabulary = payload["vocabulary"]
        axes = vocabulary["feature_axis_order"]
        if (len(axes) != vocabulary["feature_axis_count"] or len(set(axes)) != len(axes)
                or axes[:len(baseline_axes)] != baseline_axes or vocabulary["feature_axis_sha256"] != _digest(axes)):
            raise ValueError("experimental feature-axis manifest mismatch")
        normalized = deepcopy(payload)
        normalized["schema_version"] = "forte.rules.v1"
        if include_baseline:
            normalized["rules"] = deepcopy(baseline.payload["rules"]) + normalized["rules"]
            normalized["bindings"] = {**deepcopy(baseline.bindings), **normalized["bindings"]}
            normalized["event_types"] = {**deepcopy(baseline.event_types), **normalized["event_types"]}
        super().__init__(normalized, root=root, verify_archives=True)
        # Keep the marker on serialized public metadata as well: copying
        # ``catalog.payload`` must not disguise the opt-in schema as production.
        self.payload["schema_version"] = EXPERIMENTAL_SCHEMA
        self.experimental = True
        self.is_experimental = True
        self.requires_temporal_wrapper = True
        self.experimental_rule_ids = tuple(payload["experimental_rule_ids"])
        self.vocabulary = deepcopy(vocabulary)
        self.vocabulary_version = vocabulary["schema_version"]
        self._validate_experimental_temporal()

    def _validate_experimental_temporal(self) -> None:
        used: set[str] = set()
        for rule_id in self.experimental_rule_ids:
            temporal = self.bindings[rule_id]["experimental_temporal"]
            if temporal.get("schema_version") != TEMPORAL_SCHEMA:
                raise ValueError("unsupported experimental temporal schema")
            phases = temporal["ordered_phases"]
            ids = [phase["phase_id"] for phase in phases]
            if not phases or len(ids) != len(set(ids)) or temporal.get("strict_phase_order") is not True:
                raise ValueError("experimental branch requires unique strictly ordered phases")
            if temporal.get("event_phase_id") not in ids:
                raise ValueError("experimental event anchor must name a verified source phase")
            expected_interval_policy = "full_procedure_source_bounds" if self.bindings[rule_id]["is_omission"] else "event_phase_source_bounds"
            if temporal.get("candidate_interval_policy") != expected_interval_policy:
                raise ValueError("experimental event interval must derive from the applicable source phase")
            phase_facts = {key for phase in phases for key in phase["fact_ids"]}
            if not phase_facts <= self.facts.keys():
                raise ValueError("experimental phase refers to undefined fact")
            used.update(phase_facts)
            used.update(self.bindings[rule_id]["required_fact_states"])
            if not set(temporal["required_procedure_boundary_fact_ids"]) <= phase_facts:
                raise ValueError("procedure boundary fact missing from required temporal phase")
            if set(temporal["phase_procedure_roles"]) != set(ids):
                raise ValueError("every phase must have a deterministic procedure role")
            expected_roles = {"prior", "current"} if temporal["distinct_procedure_ids"] else {"current"}
            if set(temporal["phase_procedure_roles"].values()) != expected_roles:
                raise ValueError("phase procedure roles contradict distinct-procedure gate")
            tracks = temporal["required_track_kinds"]
            if set(tracks) != set(temporal["track_phase_ids"]) or set(tracks) != set(temporal["track_fact_ids"]):
                raise ValueError("track kinds require matching phases and continuity facts")
            for kind in tracks:
                if not set(temporal["track_phase_ids"][kind]) <= set(ids):
                    raise ValueError("track references unknown phase")
                fact_id = temporal["track_fact_ids"][kind]
                if self.bindings[rule_id]["required_fact_states"].get(fact_id) != "present":
                    raise ValueError("track continuity must be an affirmative global gate")
            if self.bindings[rule_id]["is_omission"] and not (
                temporal["absence_must_cover_procedure"] and temporal["counting_paths_must_cover_procedure"]
            ):
                raise ValueError("experimental count omission requires complete absence/counting-path scope")
            for key, state in self.bindings[rule_id]["required_fact_states"].items():
                self.fact_description(key, state)
        if not used <= set(self.vocabulary["feature_axis_order"]):
            raise ValueError("model vocabulary omits an experimental gate or phase fact")

    def fact_definitions(self, *, enabled_only: bool = True,
                         trusted_scene_context: str | None = None) -> list[FactDefinition]:
        # Include mandatory phase-only facts for A11's ordered context proof.
        facts = {item.fact_id: item for item in super().fact_definitions(enabled_only=enabled_only, trusted_scene_context=trusted_scene_context)}
        for rule_id in self.experimental_rule_ids:
            for phase in self.bindings[rule_id]["experimental_temporal"]["ordered_phases"]:
                for key in phase["fact_ids"]:
                    row = self.facts[key]
                    facts.setdefault(key, FactDefinition(fact_id=key, description=f"{row['question_ru']} {row['definition_ru']}", required=True))
        return [facts[key] for key in self.vocabulary["feature_axis_order"] if key in facts]

    def axis_fact_definitions(self, *, trusted_scene_context: str | None = None) -> list[FactDefinition]:
        """Full separately versioned head axis order, including baseline axes."""
        all_facts = {item.fact_id: item for item in super().fact_definitions(enabled_only=False, trusted_scene_context=trusted_scene_context)}
        return [all_facts[key] for key in self.vocabulary["feature_axis_order"]]

    def validate_axis_order(self, axis_ids: list[str] | tuple[str, ...], *, vocabulary_version: str) -> None:
        """Reject a stale/reordered head before interpreting any model output."""
        if vocabulary_version != self.vocabulary_version:
            raise ValueError("experimental model vocabulary version mismatch")
        if list(axis_ids) != self.vocabulary["feature_axis_order"]:
            raise ValueError("experimental model axis order mismatch; baseline nineteen-axis head is incompatible")


def load_experimental_catalog(path: str | Path | None = None, *, include_baseline: bool = False) -> ExperimentalRuleCatalog:
    selected = Path(path) if path is not None else EXPERIMENTAL_CATALOG_PATH
    return ExperimentalRuleCatalog(json.loads(selected.read_text(encoding="utf-8")), root=selected.parent,
                                   include_baseline=include_baseline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or inspect the explicitly optional regulatory experiment")
    parser.add_argument("--write", action="store_true", help="Write only rules/experimental_catalog.json")
    args = parser.parse_args()
    if args.write:
        payload = build_experimental_payload()
        EXPERIMENTAL_CATALOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalog = load_experimental_catalog()
    print(json.dumps({"catalog_version": catalog.version, "experimental_rules": len(catalog.experimental_rule_ids),
                      "feature_axes": catalog.vocabulary["feature_axis_count"], "baseline_axes": catalog.vocabulary["baseline_axis_count"],
                      "default_catalog_changed": False}, ensure_ascii=True))


if __name__ == "__main__":
    main()

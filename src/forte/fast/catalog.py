"""Violation types detected by the fast analyzer and their normative grounding.

Every citation below was checked against the official text of the act on
adilet.zan.kz (local copies in rules/sources and rules/research):
  * НБРК №231 — Постановление Правления Национального Банка РК от 29.11.2019 №231
    «Об утверждении Правил осуществления кассовых операций и операций по инкассации
    банкнот, монет и ценностей в банках второго уровня…» (V1900019680).
  * НБРК №14 — Постановление Правления Национального Банка РК от 24.02.2020 №14
    «Об утверждении Правил организации охраны и устройства помещений банков второго
    уровня…» (V2000020075).
  * АРРФР №53 — Постановление Правления Агентства РК по регулированию и развитию
    финансового рынка от 03.04.2026 №53 «Об утверждении Требований к обеспечению
    информационной безопасности банков…» (V2600038336).
Results are hints for a human reviewer, not legal conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass

NBK231 = ("Правила осуществления кассовых операций и операций по инкассации банкнот, монет и "
          "ценностей в банках второго уровня (утв. постановлением Правления НБРК от 29.11.2019 №231)")
NBK231_URL = "https://adilet.zan.kz/rus/docs/V1900019680"
NBK14 = ("Правила организации охраны и устройства помещений банков второго уровня "
         "(утв. постановлением Правления НБРК от 24.02.2020 №14)")
NBK14_URL = "https://adilet.zan.kz/rus/docs/V2000020075"
ARDFM53 = ("Требования к обеспечению информационной безопасности банков (утв. постановлением "
           "Правления АРРФР от 03.04.2026 №53)")
ARDFM53_URL = "https://adilet.zan.kz/rus/docs/V2600038336"


@dataclass(frozen=True)
class ViolationType:
    type_id: str
    name_ru: str
    # What the reviewer should look for (used in prompts, English for the model).
    cue_en: str
    # What correct behaviour looks like (used in prompts to avoid false alarms).
    correct_en: str
    document: str
    clause: str
    url: str
    rule_ru: str


TYPES: tuple[ViolationType, ...] = (
    ViolationType(
        "NO_CLIENT_ID",
        "Приём наличных без проверки документа, удостоверяющего личность",
        "A client hands cash to the cashier and the cashier takes it, but no ID card/passport is "
        "handed over or examined before or during the cash acceptance.",
        "The client first gives an ID card/passport; the cashier holds and looks at it, then takes cash.",
        NBK231, "пункт 19, подпункт 1)", NBK231_URL,
        "кассовый работник проводит визуальную идентификацию клиента, вносящего наличные деньги, "
        "с документом, удостоверяющим его личность"),
    ViolationType(
        "COLLECTOR_NO_DOCUMENTS",
        "Приём инкассаторской сумки без проверки документов инкассатора",
        "A cash collector brings a cash bag (sack) and the cashier accepts it without the collector "
        "showing/handing any document or ID card.",
        "The collector hands over a document/ID card together with the bag and the cashier checks it.",
        NBK231, "пункт 24 (с учётом пункта 101)", NBK231_URL,
        "приём наличных от инкассаторов осуществляется на основании препроводительного документа; "
        "кассовый работник проверяет документ, целостность и пломбирование сумки"),
    ViolationType(
        "COUNT_WITHOUT_CLIENT",
        "Пересчёт наличных в отсутствие клиента",
        "The client gives cash and leaves (walks out) before the cashier counts it; the cashier "
        "counts or feeds the cash into the counting machine after the client is gone.",
        "The client stays seated at the desk until counting is finished.",
        NBK231, "пункт 20, часть первая", NBK231_URL,
        "наличные деньги принимаются и пересчитываются кассовым работником в присутствии клиента"),
    ViolationType(
        "NO_AUTHENTICITY_CHECK",
        "Банкноты не проверены на подлинность",
        "The cashier counts banknotes by hand and finishes without ever placing them into the small "
        "banknote detector (UV/authenticity checker) on the desk.",
        "The cashier passes each banknote through the banknote detector device.",
        NBK231, "пункт 19, подпункт 2) (с учётом пункта 9)", NBK231_URL,
        "банкноты принимаются с полистным пересчётом с проверкой на подлинность и платёжность"),
    ViolationType(
        "NO_RECOUNT",
        "Приём наличных без пересчёта",
        "Bundles of cash taken out of a cash bag are put straight into the safe without being "
        "counted by hand or with the counting machine.",
        "The cashier counts the bundles (by hand or machine) before storing them.",
        NBK231, "пункт 19, подпункт 2)", NBK231_URL,
        "банкноты принимаются с полистным пересчётом"),
    ViolationType(
        "PREVIOUS_CASH_ON_DESK",
        "Наличные предыдущего клиента не убраны при обслуживании следующего",
        "After one client leaves, cash from that operation stays on the desk and the next client "
        "comes in / is served while it is still lying there.",
        "Before the next client is served the cashier puts the previous cash into the safe/locked drawer.",
        NBK231, "пункт 20, части вторая и третья", NBK231_URL,
        "на рабочем месте находятся наличные деньги только обслуживаемого клиента; все ранее принятые "
        "наличные деньги хранятся в закрывающихся на ключ устройствах"),
    ViolationType(
        "CASH_LEFT_UNATTENDED",
        "Наличные/ценности оставлены без присмотра на рабочем месте",
        "The cashier leaves the desk/room while cash, a cash bag, coin tubes or other valuables "
        "remain lying on the desk.",
        "Before leaving, everything is put into the safe and the desk is empty.",
        NBK231, "пункт 20, часть третья", NBK231_URL,
        "ранее принятые наличные деньги хранятся в специальных устройствах, закрывающихся на ключ "
        "(металлический шкаф, сейф, тележка, ящики стола, тумба)"),
    ViolationType(
        "WORKSTATION_UNLOCKED",
        "Рабочая станция не заблокирована при уходе кассира",
        "The cashier leaves the workplace and the laptop screen still shows open working content "
        "(e.g. a spreadsheet) instead of a lock screen or being closed.",
        "Before leaving, the laptop shows the lock screen (wallpaper with clock) or is closed.",
        ARDFM53, "пункт 71, подпункт 9) (ближайшая открытая норма)", ARDFM53_URL,
        "при неактивности пользователя более 30 минут информационная система автоматически завершает "
        "сеанс работы либо блокирует рабочую станцию или ноутбук с разблокировкой только при вводе "
        "аутентификационных данных; обязанность блокировать рабочую станцию при уходе с рабочего места "
        "конкретизируется внутренними документами банка"),
    ViolationType(
        "KEY_LEFT_IN_SAFE",
        "Ключ оставлен в замке сейфа",
        "The cashier walks away/leaves while the key is still inserted in the safe's lock.",
        "The cashier removes the key from the safe and takes it.",
        NBK231, "пункт 80", NBK231_URL,
        "порядок хранения ключей от сейфов на рабочем месте кассового работника обеспечивает "
        "исключение их несанкционированного использования"),
    ViolationType(
        "CASH_ROOM_DOOR_OPEN",
        "Дверь кассы не закрыта",
        "The cashier enters or works in the cash room and leaves the room door open behind them.",
        "The cashier closes the door behind them right after entering.",
        NBK14, "пункт 16, подпункт 1); пункт 8", NBK14_URL,
        "каждая касса имеет отдельный вход с закрывающейся на замок дверью; кассовый узел "
        "изолирован закрывающимся служебным входом"),
    ViolationType(
        "UNAUTHORIZED_PERSON",
        "Постороннее лицо в кассовом помещении",
        "A third person (not the cashier and not the client being served / not a collector) walks "
        "into the cash room or comes up to the cashier's desk.",
        "Only the cashier and the client being served (or a collector) are in the room.",
        NBK14, "пункт 8; пункт 16, подпункт 1)", NBK14_URL,
        "кассовый узел изолируется одним закрывающимся служебным входом, касса имеет вход с "
        "закрывающейся на замок дверью (ограничение доступа посторонних лиц)"),
    ViolationType(
        "PERSONAL_ITEMS",
        "Личные вещи, напитки или личные деньги на рабочем месте кассира",
        "The cashier keeps personal items at the cash desk: drinks a beverage, puts a coat/jacket or "
        "handbag on the desk, or handles personal money at the workplace.",
        "Only work equipment and the client's cash are on the desk.",
        NBK231, "пункт 12", NBK231_URL,
        "хранение личных вещей, наличных денег и ценностей работников банка осуществляется вне "
        "кассового узла"),
    ViolationType(
        "LEFT_DURING_SERVICE",
        "Кассир покинул рабочее место до окончания обслуживания клиента",
        "While a client is still being served at the desk, the cashier gets up and walks away "
        "from the workplace.",
        "The cashier stays at the desk until the client's operation is finished.",
        NBK231, "пункт 22", NBK231_URL,
        "кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента"),
    ViolationType(
        "CASH_HANDOVER_IRREGULAR",
        "Нарушение порядка выдачи наличных клиенту (часть купюр не передана)",
        "When returning/issuing cash, the cashier splits the stack and keeps part of the banknotes "
        "in her own hand/lap instead of giving the whole counted amount to the client.",
        "The full counted amount is handed to the client in front of them.",
        NBK231, "пункт 31, подпункт 4)", NBK231_URL,
        "кассовый работник пересчитывает подготовленные к выдаче наличные деньги в присутствии "
        "клиента и выдаёт их клиенту"),
)

BY_ID: dict[str, ViolationType] = {t.type_id: t for t in TYPES}


def citation(t: ViolationType) -> str:
    return f"{t.document}, {t.clause} ({t.url})"

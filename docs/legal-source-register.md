# Kazakhstan cash-operation source register

Reviewed on **17 October 2026**, Asia/Qyzylorda. This register contains public legal criteria only. No private recording, label, prediction, or clip description was read or included in this research.

The later [full-brief coverage reassessment](legal-coverage-review.md) stages additional public-source branches and clarifies the authorized automated local first pass. Its research sources and proposals are separate from the unchanged active runtime catalog.

The offline catalog enables **4 narrowly scoped rule branches covering 3 candidate event types**. It records **8 coverage entries**, including 5 unsupported or unresolved categories. A verified source does not establish that a particular recording satisfies the rule. Every branch requires affirmative applicability facts, original supporting frame IDs, and the complete temporal conditions specified in the catalog. Missing, unclear, or occluded observations must fail closed.

## Official source and current status

**National Bank of the Republic of Kazakhstan Board Resolution No. 231, 29 November 2019**, registered by the Ministry of Justice on 3 December 2019, No. 19680:

«Об утверждении Правил осуществления кассовых операций и операций по инкассации банкнот, монет и ценностей в банках второго уровня, филиалах банков-нерезидентов Республики Казахстан, Национальном операторе почты и юридических лицах, исключительной деятельностью которых является инкассация банкнот, монет и ценностей».

- [Official Adilet document](https://adilet.zan.kz/rus/docs/V1900019680).
- [Official public metadata and current content](https://adilet.zan.kz/api/documents/by-ngr/V1900019680?language=rus).
- [Official content API](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb/content).
- [Official version history API](https://adilet.zan.kz/api/documents/c4cd4052-63e3-475b-ba9e-93f23de13fdb/versions?page_size=100).

The official API returned document ID `c4cd4052-63e3-475b-ba9e-93f23de13fdb`, `status=upd`, `is_actual=true`; the version list independently marks this same document `is_current=true`. Its latest revision boundary is **12 July 2026**. Clause 74's footnote also gives 12 July 2026 for the relevant No. 25 amendment. Paragraph 1 establishes scope over second-tier banks, branches of nonresident banks, the postal operator and relevant collection organizations. This supports application to verified bank cash procedures, rather than to any office or person handling paper.

Two API metadata fields need care. `end_date` is the current revision boundary even while the document and history mark the record current; it was not treated as a repeal date. `adopting_organ_name_ru` says Ministry of Justice, whereas `official_name` and the normative text explicitly identify the issuing NB RK Board. The catalog uses the text and official document name for issuer attribution. The API also has inconsistent adoption-date metadata; the title and normative text establish **29 November 2019**.

The current public HTML document route returns a JavaScript shell. The full current source was retrieved directly from the official public API used by that site. The old Adilet HTML/history/info routes timed out. No third-party mirror was used to establish a rule. Source research used web search to locate the official publication and current amendments, then downloaded and checked the official text and version records locally.

## Archived provenance

`rules/sources/manifest.json` records the four public archives, full URLs, access date, SHA-256 hashes, exact byte counts, official document/version IDs and metadata caveats:

| Archive | Purpose |
| --- | --- |
| `nbk_231_current.json` | Unmodified official current metadata response, including source content |
| `nbk_231_content.json` | Unmodified official content response; primary hash linked by every RuleRecord |
| `nbk_231_versions.json` | Unmodified official version list proving the selected current record |
| `nbk_231_current.txt` | UTF-8 plain text extracted deterministically from the official HTML content for offline clause verification |

Runtime loading checks every archive hash, each rule's primary hash and source URL, and the presence of every exact clause excerpt in the archived text. A changed or missing archive raises an error. Processing does not contact the internet. Updating a source requires a new legal review and catalog hash update; it must not silently preserve an enabled mapping after a clause changes.

## Enabled rule branches

The `title` of each shared `RuleRecord` is the official document title. Reviewed Russian event labels live separately in `event_types`, and explanations state possible observable signs for human checking.

| Rule ID | Exact clause | Supported branch and necessary conditions |
| --- | --- | --- |
| `RK_NBK231_P22_RECEIPT_DEPARTURE` | Paragraph 22, within Chapter 2, paragraph/subsection 2 on receipt of cash | A cashier visibly leaves their defined workstation before the same client's receipt service finishes. Requires a confirmed bank cash context, actual receipt operation, affirmative departure, unfinished service, continuous visibility, cash on the surface and supported absence of the servicing cashier. A person merely bending, moving within the workstation, or waiting near a customer does not satisfy this branch. |
| `RK_NBK231_P20_PREVIOUS_CASH_POCKET` | Paragraph 20, second part | Previously received cash must be stored in specified key-lockable workstation devices. The branch covers only the same cash continuously tracked from a completed prior receipt into a visible clothing pocket. It excludes current-client cash before acceptance, customer-owned cash, issued cash, unknown bags and official cash cassettes. It does not infer theft, ownership, intent or permission. |
| `RK_NBK231_P19_2_RECEIPT_COUNT` | Paragraph 19, subparagraph 2 | Receipt of separate unpackaged banknotes requires leaf counting. A possible omission requires the entire completed relevant procedure, all manual and machine counting paths visible, actual client-to-cashier receipt and supported absence of counting. A short clip, invisible device or unclear view is insufficient. Sealed collection bags and the separate collection procedure in paragraphs 24–25 are excluded. |
| `RK_NBK231_P31_4_ISSUE_COUNT` | Paragraph 31, subparagraph 4, subject to paragraph 32 | Counting prepared cash in the customer's presence before issue. The branch covers only separate unpackaged banknotes, with the complete preparation/issue stage and all counting paths visible. Paragraph 32 expressly permits intact packs and banknote bundles to be issued by the amount on their packaging without leaf counting; such packages are excluded by the affirmative loose-banknote precondition. |

Paragraph 6 permits automated cash-operation devices. Therefore absence of visible hand counting cannot establish omitted counting if a machine or automatic path is hidden or uncertain.

Paragraph 20 describes devices **capable of being locked by key**. Its wording was not converted into an invented requirement that every safe/drawer door must always be closed. An open drawer alone is not an enabled violation.

The catalog's executable binding states are the complete gate, including positive scene/procedure facts, explicitly absent action facts, visibility/context facts and temporal checks. Text exceptions are considered resolved only within these narrow positive preconditions; they are not waived generally. No arbitrary minimum duration has been added because the cited clauses provide none. Action anchors use the first actual supported source frame; omission anchors use the last actual frame at the completed procedure boundary. Evidence must retain the relevant before/after context.

## Explicit gaps

| Category | Exact source context where available | Reason it cannot independently produce a grounded event |
| --- | --- | --- |
| Other person enters a workspace | Paragraph 75 | The clause concerns a vault or safe room, allowing responsible persons and persons admitted by a management document with accompaniment. An ordinary workstation is not automatically a vault. Roles, authorization documents and permitted access cannot be established from appearance. `WORKSPACE_ACCESS_BY_OTHER_PERSON` remains disabled. |
| Identity document not seen | Paragraphs 19(1), 31(1), and 7 | Identification through digital documents is permitted. An unseen paper document does not show that identification or due diligence was omitted. Bank system records and personal identity are outside this service. |
| Phone, casual clothing or a gesture | No applicable general prohibition verified | These observations cannot automatically imply a violation. No invented internal ForteBank policy was used. |
| No controller seen during cash-room work | Paragraph 42 | Requires a distinct counting/sorting/packing procedure and a controller, except during customer receipt/issue. Full room visibility, the specific procedure and role cannot be assumed. This branch is unresolved. |
| Vault storage, accounting or key control | Paragraphs 74–76 | Requires accounting-system entries, internal documents, roles and verified room classification. These are not established by visual evidence. |

The four generic taxonomy event types all have explicit full/partial/unresolved coverage entries in `rules/catalog.json`. The event type itself never supplies legal applicability. No foreign regulation or another bank's policy was mapped onto ForteBank.

## Integration and trusted scope

Import `rules.catalog.load_catalog()`; it exposes `.rules`, `.bindings`, `.facts`, `.event_types`, `.coverage_gaps`, `.enabled_rules()` and `.get_rule(rule_id, require_enabled=True)`. `load_rules(path=None)` returns the shared `RuleRecord` dictionary. Package `rules/` and its archives with the deployed service.

`.fact_definitions(enabled_only=True, trusted_scene_context=None)` returns shared `FactDefinition` objects for the 19 relevant gate facts. The only recognized trusted scene setting is `bank_cash_workstation`, supplied by an authorized local operator for bank cash footage. It confirms deployment scope and still requires the actual cash workstation to be visible. Without that setting, the `BANK_CASH_WORKSTATION` prompt explicitly requires `unclear`. Never infer regulated-bank identity from a logo, clothing or furniture, and never turn the setting into an evidence-free present fact.

`.fact_description(fact_id, state)` and `.event_type_label(event_type_id)` return reviewed phrases. `.render_description(rule, reviewed_phrases, reviewed_limitations)` rejects arbitrary model notes, fabricated clauses, altered RuleRecords and unapproved limitation text. Upstream temporal logic must establish every binding state and attach real frame IDs before explanation/evidence assembly. The renderer does not determine whether a video violates a rule.

## Validation and remaining review

Run `python -m pytest rules/tests -q` in the project environment. Checks cover official archive integrity, tampering rejection, shared-schema compatibility, taxonomy IDs, complete omission gates, explicit unobservable gaps, trusted scope handling and rejection of unreviewed generated legal text.

No private recording was legally adjudicated through this coding-agent context. Human/local-model review, evidence support on actual footage, scene attestation, applicability per recording and precision/recall remain integration and evaluation work. The source catalog is conservative legal grounding infrastructure, not a measured detection result or a legal opinion about a person.

# Синтетический учебный набор процедур кассы — 18 октября 2026

Видимое завершённое действие подтверждает шаг; отсутствие шага оценивается только при полной операции и наблюдении не менее 90%; неизвестное не является нарушением.

Набор сгенерирован независимо от банковских видео, кадров, имён файлов и разметки. Конкретные пользовательские примеры не использованы. Это учебные логические последовательности, а не новые наблюдения или нормативные заключения.

До эксперимента зафиксированы **14 управляющих параметров генерации** (список в receipt.json); ни один не выбран по результату модели. 40 групп шаблонов разделены по фиксированному seed: 32 обучающих и 8 проверочных; производные одной группы не переходят границу. На группу приходится 16 вариантов: 4 кандидата с полной видимостью, 4 корректных соседа, 4 с неполной операцией/неизвестным треком и 4 с невидимой критической областью. Кандидаты с неизвестным встроенным маршрутом проверки подлинности также имеют решение unassessable. Итого 512/128 примеров.

Вход модели содержит только requested_type, coverage и typed timeline; имена групп, split, seed и номера вариантов в prompt не попадают. t_sec — относительное время событий, не длительность ролика. Локальные object_token обозначают синтетические треки, не идентичности людей. Все поля target — закрытые перечисления и индекс события.

## Обязательные ограничения видимости

- Для пропуска пересчёта, документа, проверки подлинности или завершения выдачи нужны полная операция, observed_fraction ≥ 0.90 и видимость критических зон. Невидимый или неизвестный шаг не считается пропущенным.
- Для ключа нужны один и тот же замок и ключ, непрерывные треки, отведённая рука, наблюдаемый уход кассира и ключ после ухода. Одного прежнего вставления недостаточно.
- Для пересчёта без клиента нужны видимая связь банкнот с незавершённой клиентской операцией и непрерывная видимость клиентской зоны. Пересчёт денег, взятых из сейфа и возвращённых туда без клиентской связи, сам по себе не подтверждает этот тип.
- Неиспользованный отдельный детектор не доказывает пропуск проверки, если возможная встроенная проверка счётной машины неизвестна. Такой маршрут помечен unassessable. Для оставленных денег видны сами деньги и отсутствие ответственного после ухода.
- Частичное удержание денег само по себе не доказывает сумму, собственность, намерение или неправильность выдачи. Подтверждающие синтетические варианты показывают предложение одной отслеженной группы клиенту, его видимый жест получения, передачу части и отвод предложенного остатка. Без этой наблюдаемой связи решение unassessable.
- Для незаблокированного рабочего места рабочее содержимое экрана видно после ухода. Чёрный, закрытый или невидимый экран не доказывает нарушение.
- Неподтверждённое отсутствие шага возвращает unassessable; anchor_index=-1. Для supported индекс указывает наблюдаемое решающее событие. not_supported также имеет индекс -1.

## 40 конкретных пар последовательностей

Английские названия ниже — закрытые типы событий, не свободные описания сцен. В каждой паре одинаковые токены объектов связаны внутри одной операции. Для каждой пары также создаются два вида недостаточной видимости, описанные выше.

1. **Приём отдельных банкнот и помещение в сейф** (`NO_RECOUNT`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → banknotes_arranged → cash_stored`.
   Корректный сосед: `cash_accepted → banknotes_arranged → machine_count_completed → cash_stored`.

2. **Вскрытие инкассаторского мешка перед хранением** (`NO_RECOUNT`, обучение).
   Кандидат с обязательными защитными условиями: `bag_opened → bundles_removed → cash_stored`.
   Корректный сосед: `bag_opened → bundles_removed → manual_count_completed → cash_stored`.

3. **Перенос банкнот с лотка в кассету сейфа** (`NO_RECOUNT`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → cash_on_tray → cashette_loaded → cash_stored`.
   Корректный сосед: `cash_accepted → cash_on_tray → machine_count_completed → cashette_loaded → cash_stored`.

4. **Упаковка принятых банкнот в конверт** (`NO_RECOUNT`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → envelope_loaded → cash_stored`.
   Корректный сосед: `cash_accepted → manual_count_completed → envelope_loaded → cash_stored`.

5. **Несколько пачек после вскрытия упаковки** (`NO_RECOUNT`, обучение).
   Кандидат с обязательными защитными условиями: `package_opened → bundles_separated → all_bundles_stored`.
   Корректный сосед: `package_opened → bundles_separated → all_bundles_count_completed → all_bundles_stored`.

6. **Промежуточный лоток перед помещением в сейф** (`NO_RECOUNT`, проверка).
   Кандидат с обязательными защитными условиями: `cash_accepted → cash_moved_to_staging_tray → cash_stored`.
   Корректный сосед: `cash_accepted → cash_moved_to_staging_tray → manual_count_completed → cash_stored`.

7. **Закрытая дверь сейфа, кассир ушёл от сейфа** (`KEY_LEFT_IN_SAFE`, обучение).
   Кандидат с обязательными защитными условиями: `key_inserted → safe_opened → safe_closed → hand_withdrawn → cashier_left_safe_zone → key_visible_after_departure`.
   Корректный сосед: `key_inserted → safe_opened → safe_closed → key_removed → hand_withdrawn → cashier_left_safe_zone → lock_observed_empty`.

8. **Ключ остаётся после выхода кассира из комнаты** (`KEY_LEFT_IN_SAFE`, проверка).
   Кандидат с обязательными защитными условиями: `key_inserted → safe_closed → hand_withdrawn → cashier_left_room → key_visible_after_departure`.
   Корректный сосед: `key_inserted → safe_closed → key_removed → hand_withdrawn → cashier_left_room → lock_observed_empty`.

9. **Повторное закрытие сейфа после второй операции** (`KEY_LEFT_IN_SAFE`, обучение).
   Кандидат с обязательными защитными условиями: `key_inserted → safe_opened → safe_closed → safe_reopened → safe_closed → hand_withdrawn → cashier_left_safe_zone → key_visible_after_departure`.
   Корректный сосед: `key_inserted → safe_opened → safe_closed → safe_reopened → safe_closed → key_removed → hand_withdrawn → cashier_left_safe_zone → lock_observed_empty`.

10. **Кассир вернулся к столу после сейфа** (`KEY_LEFT_IN_SAFE`, обучение).
   Кандидат с обязательными защитными условиями: `key_inserted → safe_closed → hand_withdrawn → cashier_returned_to_desk → key_visible_after_departure`.
   Корректный сосед: `key_inserted → safe_closed → key_removed → hand_withdrawn → cashier_returned_to_desk → lock_observed_empty`.

11. **Уборка лотка и отход от сейфа** (`KEY_LEFT_IN_SAFE`, обучение).
   Кандидат с обязательными защитными условиями: `key_inserted → safe_closed → tray_removed → hand_withdrawn → cashier_left_safe_zone → key_visible_after_departure`.
   Корректный сосед: `key_inserted → safe_closed → key_removed → tray_removed → hand_withdrawn → cashier_left_safe_zone → lock_observed_empty`.

12. **Окончание работы у сейфа с контролем замка** (`KEY_LEFT_IN_SAFE`, обучение).
   Кандидат с обязательными защитными условиями: `safe_opened → key_inserted → safe_closed → hand_withdrawn → cashier_left_safe_zone → key_visible_after_departure`.
   Корректный сосед: `safe_opened → key_inserted → safe_closed → key_removed → hand_withdrawn → cashier_left_safe_zone → lock_observed_empty`.

13. **Пересчёт начинается после ухода клиента** (`COUNT_WITHOUT_CLIENT`, обучение).
   Кандидат с обязательными защитными условиями: `client_left → client_region_observed_empty → count_started → count_completed`.
   Корректный сосед: `client_present → count_started → count_completed → client_left`.

14. **Ручной пересчёт без клиента у стола** (`COUNT_WITHOUT_CLIENT`, обучение).
   Кандидат с обязательными защитными условиями: `client_left → client_region_observed_empty → manual_count_started → count_completed`.
   Корректный сосед: `client_present → manual_count_started → count_completed → client_left`.

15. **Клиент отсутствует весь машинный пересчёт** (`COUNT_WITHOUT_CLIENT`, обучение).
   Кандидат с обязательными защитными условиями: `client_region_observed_empty → cash_loaded_in_counter → count_started → count_completed`.
   Корректный сосед: `client_present → cash_loaded_in_counter → count_started → count_completed`.

16. **Незавершённая операция клиента против пересчёта из сейфа** (`COUNT_WITHOUT_CLIENT`, проверка).
   Кандидат с обязательными защитными условиями: `client_present → initial_count_completed → client_left → client_region_observed_empty → count_started → count_completed`.
   Корректный сосед: `cash_removed_from_safe_for_count → client_region_observed_empty → count_started → count_completed → cash_returned_to_same_safe`.

17. **Передача наличных без предъявления документа** (`NO_CLIENT_ID`, обучение).
   Кандидат с обязательными защитными условиями: `client_present → cash_accepted → service_completed`.
   Корректный сосед: `client_present → document_presented → document_checked → cash_accepted → service_completed`.

18. **Внесение наличных после посадки клиента** (`NO_CLIENT_ID`, обучение).
   Кандидат с обязательными защитными условиями: `client_sat → cash_accepted → cash_processed → service_completed`.
   Корректный сосед: `client_sat → document_presented → document_checked → cash_accepted → cash_processed → service_completed`.

19. **Приём конверта с деньгами от клиента** (`NO_CLIENT_ID`, проверка).
   Кандидат с обязательными защитными условиями: `client_present → envelope_opened → cash_accepted → service_completed`.
   Корректный сосед: `client_present → document_presented → document_checked → envelope_opened → cash_accepted → service_completed`.

20. **Вторая операция нового клиента** (`NO_CLIENT_ID`, проверка).
   Кандидат с обязательными защитными условиями: `previous_client_left → new_client_present → cash_accepted → service_completed`.
   Корректный сосед: `previous_client_left → new_client_present → document_presented → document_checked → cash_accepted → service_completed`.

21. **Пересчитанные банкноты сразу убраны** (`NO_AUTHENTICITY_CHECK`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → machine_count_completed → cash_stored`.
   Корректный сосед: `cash_accepted → machine_count_completed → authenticity_check_completed → cash_stored`.

22. **Ручной пересчёт перед хранением** (`NO_AUTHENTICITY_CHECK`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → manual_count_completed → cash_stored`.
   Корректный сосед: `cash_accepted → manual_count_completed → authenticity_check_completed → cash_stored`.

23. **Обработка нескольких пачек** (`NO_AUTHENTICITY_CHECK`, обучение).
   Кандидат с обязательными защитными условиями: `bundles_removed → all_bundles_count_completed → all_bundles_stored`.
   Корректный сосед: `bundles_removed → all_bundles_count_completed → all_bundles_authenticity_checked → all_bundles_stored`.

24. **Закрытие операции после приёма наличных** (`NO_AUTHENTICITY_CHECK`, обучение).
   Кандидат с обязательными защитными условиями: `cash_accepted → count_completed → service_completed`.
   Корректный сосед: `cash_accepted → count_completed → authenticity_check_completed → service_completed`.

25. **Наличные на столе после выхода кассира** (`CASH_LEFT_UNATTENDED`, обучение).
   Кандидат с обязательными защитными условиями: `cash_on_desk → cashier_left_room → responsible_region_observed_empty → cash_visible_after_departure`.
   Корректный сосед: `cash_on_desk → cash_secured → cashier_left_room → desk_observed_clear`.

26. **Кассир уходит, деньги остаются на лотке** (`CASH_LEFT_UNATTENDED`, обучение).
   Кандидат с обязательными защитными условиями: `cash_on_tray → cashier_left_desk → responsible_region_observed_empty → cash_visible_after_departure`.
   Корректный сосед: `cash_on_tray → cash_secured → cashier_left_desk → desk_observed_clear`.

27. **Пачка у счётной машины без ответственного** (`CASH_LEFT_UNATTENDED`, обучение).
   Кандидат с обязательными защитными условиями: `cash_beside_counter → cashier_left_room → responsible_region_observed_empty → cash_visible_after_departure`.
   Корректный сосед: `cash_beside_counter → cash_secured → cashier_left_room → desk_observed_clear`.

28. **После клиента кассир покидает рабочую зону** (`CASH_LEFT_UNATTENDED`, проверка).
   Кандидат с обязательными защитными условиями: `client_left → cash_on_desk → cashier_left_desk → responsible_region_observed_empty → cash_visible_after_departure`.
   Корректный сосед: `client_left → cash_on_desk → cash_secured → cashier_left_desk → desk_observed_clear`.

29. **Приём мешка инкассатора без документов** (`COLLECTOR_NO_DOCUMENTS`, проверка).
   Кандидат с обязательными защитными условиями: `collector_entered → bag_presented → bag_accepted → collector_service_completed`.
   Корректный сосед: `collector_entered → collector_document_presented → collector_document_checked → bag_presented → bag_accepted → collector_service_completed`.

30. **Сумка передана после подхода к столу** (`COLLECTOR_NO_DOCUMENTS`, обучение).
   Кандидат с обязательными защитными условиями: `collector_at_desk → bag_presented → bag_accepted → collector_service_completed`.
   Корректный сосед: `collector_at_desk → collector_document_presented → collector_document_checked → bag_presented → bag_accepted → collector_service_completed`.

31. **Приём второй сумки в отдельном визите** (`COLLECTOR_NO_DOCUMENTS`, обучение).
   Кандидат с обязательными защитными условиями: `previous_collector_left → collector_entered → bag_presented → bag_accepted → collector_service_completed`.
   Корректный сосед: `previous_collector_left → collector_entered → collector_document_presented → collector_document_checked → bag_presented → bag_accepted → collector_service_completed`.

32. **Открытие принятого мешка** (`COLLECTOR_NO_DOCUMENTS`, обучение).
   Кандидат с обязательными защитными условиями: `collector_entered → bag_accepted → bag_opened → collector_service_completed`.
   Корректный сосед: `collector_entered → collector_document_presented → collector_document_checked → bag_accepted → bag_opened → collector_service_completed`.

33. **После разделения часть банкнот удержана** (`CASH_HANDOVER_IRREGULAR`, обучение).
   Кандидат с обязательными защитными условиями: `cash_prepared → cash_split → part_handed_to_client → part_retained → client_left → service_ended_with_retained_cash`.
   Корректный сосед: `cash_prepared → cash_split → part_handed_to_client → remaining_part_handed_to_client → client_left → service_completed`.

34. **Часть суммы остаётся в лотке до ухода клиента** (`CASH_HANDOVER_IRREGULAR`, проверка).
   Кандидат с обязательными защитными условиями: `cash_on_tray → cash_split → part_handed_to_client → part_retained_on_tray → client_left → service_ended_with_retained_cash`.
   Корректный сосед: `cash_on_tray → cash_split → part_handed_to_client → remaining_part_handed_to_client → client_left → service_completed`.

35. **После пересчёта передана только одна часть** (`CASH_HANDOVER_IRREGULAR`, обучение).
   Кандидат с обязательными защитными условиями: `count_completed → cash_split → part_handed_to_client → part_retained → client_left → service_ended_with_retained_cash`.
   Корректный сосед: `count_completed → cash_split → part_handed_to_client → remaining_part_handed_to_client → client_left → service_completed`.

36. **Передача двух пачек завершается удержанием одной** (`CASH_HANDOVER_IRREGULAR`, обучение).
   Кандидат с обязательными защитными условиями: `bundles_prepared → cash_split → part_handed_to_client → part_retained → client_left → service_ended_with_retained_cash`.
   Корректный сосед: `bundles_prepared → cash_split → part_handed_to_client → remaining_part_handed_to_client → client_left → service_completed`.

37. **Рабочее приложение после выхода из комнаты** (`WORKSTATION_UNLOCKED`, обучение).
   Кандидат с обязательными защитными условиями: `working_screen_visible → cashier_left_room → working_screen_visible_after_departure`.
   Корректный сосед: `working_screen_visible → screen_locked → cashier_left_room → lock_screen_visible_after_departure`.

38. **Отход от стола при рабочем экране** (`WORKSTATION_UNLOCKED`, обучение).
   Кандидат с обязательными защитными условиями: `working_screen_visible → cashier_left_desk → working_screen_visible_after_departure`.
   Корректный сосед: `working_screen_visible → screen_locked → cashier_left_desk → lock_screen_visible_after_departure`.

39. **Уход после окончания обслуживания** (`WORKSTATION_UNLOCKED`, обучение).
   Кандидат с обязательными защитными условиями: `service_completed → working_screen_visible → cashier_left_room → working_screen_visible_after_departure`.
   Корректный сосед: `service_completed → working_screen_visible → screen_locked → cashier_left_room → lock_screen_visible_after_departure`.

40. **Рабочий экран остаётся после возврата к сейфу** (`WORKSTATION_UNLOCKED`, обучение).
   Кандидат с обязательными защитными условиями: `working_screen_visible → cashier_left_desk → cashier_at_safe → working_screen_visible_after_departure`.
   Корректный сосед: `working_screen_visible → screen_locked → cashier_left_desk → cashier_at_safe → lock_screen_visible_after_departure`.

## Пределы результата

Это новый синтетический holdout по шаблонам, но не банковская оценка и не проверка зрения. Он не подтверждает качество восприятия пересчёта, ключей, документов или экранов на видео. Улучшение текстовой логики не является результатом 37/40. Синтетические примеры не заменяют отрицательный гейт на 19 банковских отрицательных роликах и не разрешают включение новых типов.

В наборе нет юридических цитат, аудио, лиц, личностей, сумм, OCR, кадров или организаторских меток. Допускаются только существующие catalog type_id. Выбор checkpoint по банковской разметке запрещён.

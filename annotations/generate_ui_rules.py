"""Package the generic enabled rule snapshot into static review UI HTML."""

import json
from pathlib import Path
import re


def main() -> None:
    from rules.catalog import load_catalog
    catalog = load_catalog()
    payload = {"catalog_version": catalog.version, "rules": [{"rule_id": rule.rule_id, "title": rule.title, "clause": rule.clause, "event_type_id": catalog.bindings[rule.rule_id]["event_type_id"], "required_fact_states": catalog.bindings[rule.rule_id]["required_fact_states"]} for rule in catalog.enabled_rules()]}
    serialized = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("&", "\\u0026")
    path = Path(__file__).with_name("ui") / "index.html"
    original = path.read_text(encoding="utf-8")
    result, count = re.subn(r'(<script id="review-rule-catalog" type="application/json">).*?(</script>)', lambda match: match[1] + serialized + match[2], original, flags=re.S)
    if count != 1:
        raise ValueError("UI rule snapshot marker must be unique")
    path.write_text(result, encoding="utf-8")
    print(json.dumps({"enabled_rule_choices": len(payload["rules"])}))


if __name__ == "__main__":
    main()

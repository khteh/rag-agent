import re, logging
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown
from src.models.EmailModel import EmailModel

class RobustEmailModelParser(BaseOutputParser[EmailModel]):
    """
    Robust parser for EmailModel — handles JSON output AND markdown table output.
    with_structured_output raises OutputParserException before Pydantic ever
    sees non-JSON text, so this parser intercepts at the string level.
    """
    _EMAIL_KEY_MAP: dict[str, str] = {
        "date": "date_str",
        "date_str": "date_str",
        "from_name": "name",
        "name": "name",
        "from_email": "email",
        "email": "email",
        "phone_number": "phone",
        "phone": "phone",
        "project_id": "project_id",
        "site_location": "site_location",
        "violation_types": "violation_types",
        "violation_type": "violation_types",
        "required_changes": "required_changes",
        "compliance_deadline": "compliance_deadline_str",
        "compliance_deadline_str": "compliance_deadline_str",
        "max_potential_fine": "max_potential_fine",
        "maximum_potential_fine": "max_potential_fine",
    }
    # Values the LLM uses to indicate a field is absent — treat as None.
    _ABSENT_VALUES = frozenset({
        "", "-", "—", "null", "none", "n/a", "na",
        "not present in the email",
        "not mentioned in the email",
        "not provided",
        "not specified",
    })
    def parse(self, text: str) -> EmailModel:
        """
        Parses EmailModel from either a JSON string or a markdown table.
        gpt-oss ignores json_schema mode and emits a markdown table; this parser
        handles both formats so the chain never raises OutputParserException.
        """
        logging.info(f"\n=== {self.parse.__name__} ===")
        # 1. Try standard JSON / JSON-in-markdown-fence path.
        try:
            data = parse_json_markdown(text)
            return EmailModel.model_validate(data)
        except Exception as e:
            logging.critical(f"{self.parse.__name__} Exception! {e}")

        # 2. Parse markdown table rows into a dict.
        data = self._parse_markdown_table_to_email_dict(text)
        if data:
            return EmailModel.model_validate(data)

        msg = f"{self.parse.__name__} Could not parse EmailModel from LLM output:\n{text}"
        logging.error(msg)
        raise OutputParserException(msg)

    @property
    def _type(self) -> str:
        return "robust_email_model"
    
    def _parse_markdown_table_to_email_dict(self, text: str) -> dict:
        result: dict = {}
        for line in text.splitlines():
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) < 2:
                continue
            raw_key = re.sub(r"\*+", "", parts[0]).strip().lower().replace(" ", "_")
            # Join remaining pipe-parts in case the value itself contained "|"
            raw_val = "|".join(parts[1:]).strip()
            # Strip markdown emphasis and replace <br> separators with " - "
            raw_val = re.sub(r"\*+|_+", "", raw_val).strip()
            raw_val = re.sub(r"<br\s*/?>", " - ", raw_val, flags=re.I).strip()
            if raw_val.lower() in self._ABSENT_VALUES:
                continue
            canonical = self._EMAIL_KEY_MAP.get(raw_key)
            if canonical:
                result[canonical] = raw_val
        return result
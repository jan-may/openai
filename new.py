import os
import json
from typing import Dict, Any
from openai import OpenAI

class UrhebertAnalyzer:
    """
    Analyzer for determining the organization type (urheber) of a domain using OpenAI's Responses API with web search.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, document: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Analyze the given document to determine the organization type.

        Args:
            document (Dict[str, Any]): The document containing domain information.

        Returns:
            Dict[str, Any]: Analysis results including the organization type and related metadata.
        """
        domain_info = document.get("domain_info", {})
        domain = domain_info.get("domain", "")
        tld = domain_info.get("tld", "")
        url = document.get("url", "")

        if not domain or not tld:
            return {
                "urheber": "unbekannt",
                "confidence": 0,
                "method": "failure",
                "reason": "Missing domain or TLD information."
            }

        full_domain = f"{domain}.{tld}"

        # Construct the prompt for classification
        prompt = f"""
Recherchiere den Betreiber der Website/Domain: {full_domain} ({url}).

Aufgabe: Bestimme, welcher Kategorie der Betreiber der Website angehört.

Kategorien:
- "staatlich" (Behörde, Ministerium, staatliche Einrichtung)
- "nicht staatliche Hilfsorganisation" (NGO, Hilfswerk, gemeinnützige Organisation)
- "sonstige Vereine" (Vereine, die keine Hilfsorganisationen sind)
- "Organisationen" (internationale Organisationen, Verbände)
- "Gemeinschaften" (informelle Gruppen, Communities, Netzwerke)
- "Unternehmen" (kommerzielle Unternehmen, GmbH, AG, etc.)
- "Privatperson" (von Einzelpersonen betriebene Webseiten)

Bitte gib die Kategorie des Betreibers an, eine kurze Begründung und nenne die verwendeten Quellen.
Antworte im folgenden JSON-Format:
{{
  "urheber": "KATEGORIE",
  "begründung": "Kurze Begründung für die Einordnung",
  "quellen": ["Quelle 1", "Quelle 2"]
}}
"""

        try:
            # Use the Responses API with the web_search tool
            response = self.client.responses.create(
                model="gpt-4o",
                input=prompt,
                tools=[{"type": "web_search"}]
            )

            # Extract the output text
            output_text = response.output_text

            # Parse the JSON response
            result = json.loads(output_text)

            # Add additional metadata
            result.update({
                "method": "responses_api_web_search",
                "confidence": 0.9,
                "domain": full_domain
            })

            return result

        except Exception as e:
            return {
                "urheber": "unbekannt",
                "confidence": 0,
                "method": "error",
                "error": str(e),
                "domain": full_domain
            }

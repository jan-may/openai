"""
Analyzer for determining the organization type of a domain
"""
from typing import Dict, Any, Optional
from analyzers.base_analyzer import BaseAnalyzer
import json

class UrhebertAnalyzer(BaseAnalyzer):
    """Analyzer for determining the organization type of a domain"""
    
    def analyze(self, document: Dict, **kwargs) -> Dict:
        """
        Determine the organization type (urheber) for a domain.
        
        First checks the database for existing entries with the same domain.
        If none found, uses web search to research the domain.
        
        Args:
            document: Document dictionary
            **kwargs: Additional arguments including MongoDB collection
        
        Returns:
            Organization type analysis results
        """
        url = document.get('url', '')
        collection = kwargs.get("collection", None)
        
        # Extract the domain from the URL (use domain_info if present in the document)
        domain_info = document.get("domain_info", {})
        domain = domain_info.get("domain", "")
        tld = domain_info.get("tld", "")
        
        if not domain or not tld:
            return {"urheber": "unbekannt", "confidence": 0, "method": "failure"}
        
        full_domain = f"{domain}.{tld}"
        
        # 1. Check MongoDB for existing entries with the same domain
        if collection is not None:
            existing_urheber = self._check_database_for_domain(collection, full_domain)
            if existing_urheber:
                return {
                    "urheber": existing_urheber,
                    "confidence": 1.0,
                    "method": "database",
                    "domain": full_domain
                }
        
        # 2. Use web search to determine the organization type
        return self._research_domain_with_web_search(full_domain, url)
    
    def _check_database_for_domain(self, collection, domain: str) -> str:
        """Check if there are any documents with the same domain that have an urheber field"""
        try:
            # Query for documents containing the domain in their URL
            domain_query = {"domain": {"$regex": domain, "$options": "i"}}
            # Look for documents with an urheber field
            urheber_query = {"urheber": {"$exists": True, "$ne": ""}}
            
            # Combine queries
            query = {"$and": [domain_query, urheber_query]}
            
            # Find one matching document
            result = collection.find_one(query, {"urheber": 1, "_id": 0})
            
            if result and "urheber" in result:
                print(f"Found existing urheber classification for domain {domain}: {result['urheber']}")
                return result["urheber"]
            else:
                print(f"No existing urheber classification found for domain {domain}")
        except Exception as e:
            print(f"Error querying MongoDB for domain {domain}: {e}")
        
        return None
    
    def _research_domain_with_web_search(self, domain: str, url: str) -> Dict:
        """
        Use OpenAI with web search to research the domain and determine organization type.
        
        Args:
            domain: The domain to research (e.g., example.com)
            url: The full URL for additional context
            
        Returns:
            Organization type analysis results
        """
        print(f"Researching domain {domain} using web search...")
        
        # First, define the web search function
        web_search_function = {
            "name": "web_search",
            "description": "Search the web for information about a domain or website",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string", 
                        "description": "The search query to use"
                    }
                },
                "required": ["query"]
            }
        }
        
        # Create a search query prompt
        search_query = f"Wer betreibt die Website {domain} oder {url}? Ist es eine staatliche Einrichtung, Organisation, Unternehmen oder Privatperson?"
        
        try:
            # First, execute a search using the web_search function
            search_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You search the web for information and provide factual information only."},
                    {"role": "user", "content": search_query}
                ],
                tools=[{"type": "function", "function": web_search_function}],
                tool_choice={"type": "function", "function": {"name": "web_search"}}
            )
            
            # Check if there's a tool call in the response
            search_message = search_response.choices[0].message
            
            # Extract the search results
            if hasattr(search_message, 'tool_calls') and search_message.tool_calls:
                tool_call = search_message.tool_calls[0]
                search_args = json.loads(tool_call.function.arguments)
                search_results = f"Search query: {search_args.get('query', search_query)}"
                
                # Now get the search results by submitting the tool call
                results_response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You search the web for information and provide factual information only."},
                        {"role": "user", "content": search_query},
                        search_message,
                        {"role": "tool", "tool_call_id": tool_call.id, "name": "web_search", "content": "Search results for the domain..."}
                    ]
                )
                
                search_results = results_response.choices[0].message.content
            else:
                search_results = "No search results available."
            
            # Now classify based on search results
            classification_prompt = f"""Basierend auf den folgenden Suchergebnissen, klassifiziere den Betreiber der Domain {domain} in GENAU EINE dieser Kategorien:
            
- "staatlich" (Behörde, Ministerium, staatliche Einrichtung)
- "nicht staatliche Hilfsorganisation" (NGO, Hilfswerk, gemeinnützige Organisation)
- "sonstige Vereine" (Vereine, die keine Hilfsorganisationen sind)
- "Organisationen" (internationale Organisationen, Verbände)
- "Gemeinschaften" (informelle Gruppen, Communities, Netzwerke)
- "Unternehmen" (kommerzielle Unternehmen, GmbH, AG, etc.)
- "Privatperson" (von Einzelpersonen betriebene Webseiten)

Suchergebnisse:
{search_results}

URL: {url}

Antworte im JSON-Format:
{{
  "urheber": "KATEGORIE",
  "begründung": "Kurze Begründung für die Einordnung",
  "quellen": ["Quelle 1", "Quelle 2"]
}}"""
            
            classification_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You analyze website operators and classify them into categories."},
                    {"role": "user", "content": classification_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Process and return results
            result = self._process_response(classification_response.choices[0].message.content)
            result["method"] = "web_search"
            result["confidence"] = 0.8  # Estimated confidence for web search
            result["domain"] = domain
            
            print(f"Web search classification for {domain}: {result.get('urheber', 'unknown')}")
            return result
            
        except Exception as e:
            print(f"Error using web search for domain {domain}: {e}")
            return {
                "urheber": "unbekannt", 
                "confidence": 0, 
                "method": "error", 
                "error": str(e),
                "domain": domain
            }
            
    def _create_urheber_prompt(self, domain: str, url: str) -> str:
        """Create prompt for organization type analysis"""
        return f"""Recherchiere den Betreiber der Website/Domain: {domain} ({url}).

Aufgabe: Bestimme, welcher Kategorie der Betreiber der Website angehört. 

Suche nach:
1. Informationen über den Webseitenbetreiber (Impressum, About, Legal, Contact pages)
2. Organisationsform (GmbH, AG, e.V., Behörde, etc.)
3. Staatliche Verbindungen oder Funktion
4. Nonprofit-Status oder gemeinnütziger Zweck
5. Ob es sich um eine Privatperson handelt

Klassifiziere den Betreiber in GENAU EINE dieser Kategorien:
- "staatlich" (Behörde, Ministerium, staatliche Einrichtung)
- "nicht staatliche Hilfsorganisation" (NGO, Hilfswerk, gemeinnützige Organisation)
- "sonstige Vereine" (Vereine, die keine Hilfsorganisationen sind)
- "Organisationen" (internationale Organisationen, Verbände)
- "Gemeinschaften" (informelle Gruppen, Communities, Netzwerke)
- "Unternehmen" (kommerzielle Unternehmen, GmbH, AG, etc.)
- "Privatperson" (von Einzelpersonen betriebene Webseiten)

Begründe deine Einordnung und gib deine Quellen an.
Antworte im JSON-Format.
"""

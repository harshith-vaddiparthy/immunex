"""
Literature Scanner Agent

Continuously mines PubMed and bioRxiv for drug-innate immunity associations.
Extracts structured drug-pathway-effect triples using LLM-powered NLP.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Innate immune pathways and their key terms for targeted search
INNATE_IMMUNE_PATHWAYS = {
    "tlr_signaling": {
        "name": "Toll-like Receptor Signaling",
        "targets": ["TLR2", "TLR4", "TLR7", "TLR9", "MyD88", "TRIF", "IRAK4"],
        "search_terms": [
            "toll-like receptor", "TLR signaling", "TLR agonist", "TLR antagonist",
            "MyD88", "TRIF pathway",
        ],
    },
    "inflammasome": {
        "name": "Inflammasome Activation",
        "targets": ["NLRP3", "NLRC4", "AIM2", "ASC", "caspase-1"],
        "search_terms": [
            "NLRP3 inflammasome", "inflammasome activation", "IL-1beta",
            "pyroptosis", "caspase-1 activation",
        ],
    },
    "cgas_sting": {
        "name": "cGAS-STING Pathway",
        "targets": ["cGAS", "STING", "TBK1", "IRF3"],
        "search_terms": [
            "cGAS-STING", "STING pathway", "cyclic GMP-AMP", "type I interferon",
            "DNA sensing innate",
        ],
    },
    "nfkb": {
        "name": "NF-κB Signaling",
        "targets": ["IKKβ", "RelA", "NEMO", "IκBα", "p65"],
        "search_terms": [
            "NF-kB signaling", "NF-kappaB", "IKK inhibitor", "inflammatory signaling",
        ],
    },
    "jak_stat": {
        "name": "JAK-STAT Pathway",
        "targets": ["JAK1", "JAK2", "TYK2", "STAT1", "STAT3"],
        "search_terms": [
            "JAK-STAT signaling", "JAK inhibitor", "cytokine signaling",
            "interferon signaling JAK",
        ],
    },
    "trained_immunity": {
        "name": "Trained Immunity",
        "targets": ["mTOR", "HIF-1α", "KDM5", "SET7"],
        "search_terms": [
            "trained immunity", "innate immune memory", "epigenetic reprogramming monocyte",
            "beta-glucan training", "BCG trained immunity",
        ],
    },
    "complement": {
        "name": "Complement System",
        "targets": ["C3", "C5", "Factor D", "Factor B", "MASP-2"],
        "search_terms": [
            "complement activation", "complement inhibitor", "C5a receptor",
            "complement pathway drug",
        ],
    },
}


class PubMedArticle(BaseModel):
    """Structured representation of a PubMed article."""

    pmid: str
    title: str
    abstract: str
    authors: list[str]
    journal: str
    pub_date: str
    doi: Optional[str] = None
    mesh_terms: list[str] = []
    keywords: list[str] = []


@dataclass
class DrugPathwayAssociation:
    """A structured drug-pathway-effect triple extracted from literature."""

    drug_name: str
    pathway: str
    effect: str  # "activator", "inhibitor", "modulator", "unknown"
    target_protein: Optional[str] = None
    confidence: float = 0.0  # 0-1
    source_pmid: str = ""
    source_title: str = ""
    extraction_method: str = "llm"
    extracted_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LiteratureScanner:
    """
    Agent 1: Literature Scanner

    Scans PubMed for articles describing drug effects on innate immune pathways.
    Uses the NCBI E-utilities API for search and retrieval, then optionally
    uses an LLM to extract structured drug-pathway-effect associations.
    """

    PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(
        self,
        ncbi_api_key: Optional[str] = None,
        ncbi_email: Optional[str] = None,
        llm_client=None,
    ):
        self.api_key = ncbi_api_key
        self.email = ncbi_email
        self.llm_client = llm_client
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _build_repurposing_query(self, pathway_key: str) -> str:
        """Build a PubMed query for drug repurposing in a specific innate immune pathway."""
        pathway = INNATE_IMMUNE_PATHWAYS[pathway_key]
        pathway_terms = " OR ".join(f'"{t}"' for t in pathway["search_terms"])
        drug_terms = '"drug repurposing" OR "drug repositioning" OR "off-target effect" OR "repurposed"'
        return f"({pathway_terms}) AND ({drug_terms}) AND (innate immunity OR innate immune)"

    async def search_pubmed(
        self, query: str, max_results: int = 100, min_date: str = "2020/01/01"
    ) -> list[str]:
        """Search PubMed and return a list of PMIDs."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "mindate": min_date,
            "datetype": "pdat",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with self.session.get(self.PUBMED_SEARCH_URL, params=params) as resp:
            data = await resp.json()
            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])
            total = result.get("count", "0")
            logger.info(f"PubMed search returned {total} total results, fetched {len(pmids)} IDs")
            return pmids

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch article details for a list of PMIDs."""
        if not pmids:
            return []
        if not self.session:
            self.session = aiohttp.ClientSession()

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
            "rettype": "abstract",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        async with self.session.get(self.PUBMED_FETCH_URL, params=params) as resp:
            xml_text = await resp.text()
            return self._parse_pubmed_xml(xml_text)

    def _parse_pubmed_xml(self, xml_text: str) -> list[PubMedArticle]:
        """Parse PubMed XML response into structured articles."""
        from xml.etree import ElementTree as ET

        articles = []
        try:
            root = ET.fromstring(xml_text)
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    medline = article_elem.find("MedlineCitation")
                    if medline is None:
                        continue

                    pmid_elem = medline.find("PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    article = medline.find("Article")
                    if article is None:
                        continue

                    title_elem = article.find("ArticleTitle")
                    title = title_elem.text if title_elem is not None else ""

                    # Extract abstract
                    abstract_parts = []
                    abstract_elem = article.find("Abstract")
                    if abstract_elem is not None:
                        for text_elem in abstract_elem.findall("AbstractText"):
                            label = text_elem.get("Label", "")
                            text = text_elem.text or ""
                            if label:
                                abstract_parts.append(f"{label}: {text}")
                            else:
                                abstract_parts.append(text)
                    abstract = " ".join(abstract_parts)

                    # Authors
                    authors = []
                    author_list = article.find("AuthorList")
                    if author_list is not None:
                        for author in author_list.findall("Author"):
                            last = author.find("LastName")
                            first = author.find("ForeName")
                            if last is not None and first is not None:
                                authors.append(f"{first.text} {last.text}")

                    # Journal
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    # Date
                    pub_date_elem = article.find(".//PubDate")
                    pub_date = ""
                    if pub_date_elem is not None:
                        year = pub_date_elem.find("Year")
                        month = pub_date_elem.find("Month")
                        pub_date = f"{year.text if year is not None else ''}-{month.text if month is not None else '01'}"

                    # DOI
                    doi = None
                    for id_elem in article_elem.findall(".//ArticleId"):
                        if id_elem.get("IdType") == "doi":
                            doi = id_elem.text

                    # MeSH terms
                    mesh_terms = []
                    mesh_list = medline.find("MeshHeadingList")
                    if mesh_list is not None:
                        for mesh in mesh_list.findall("MeshHeading"):
                            desc = mesh.find("DescriptorName")
                            if desc is not None:
                                mesh_terms.append(desc.text)

                    articles.append(
                        PubMedArticle(
                            pmid=pmid,
                            title=title or "",
                            abstract=abstract,
                            authors=authors,
                            journal=journal or "",
                            pub_date=pub_date,
                            doi=doi,
                            mesh_terms=mesh_terms,
                        )
                    )
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")

        return articles

    async def extract_associations_llm(
        self, article: PubMedArticle
    ) -> list[DrugPathwayAssociation]:
        """Use LLM to extract drug-pathway-effect triples from an article."""
        if not self.llm_client:
            logger.warning("No LLM client configured, skipping extraction")
            return []

        prompt = f"""Analyze this biomedical article and extract any drug-innate immunity pathway associations.

Title: {article.title}
Abstract: {article.abstract}

For each association found, provide:
1. drug_name: The specific drug or compound
2. pathway: Which innate immune pathway (TLR, inflammasome, cGAS-STING, NF-kB, JAK-STAT, trained_immunity, complement)
3. effect: "activator", "inhibitor", "modulator", or "unknown"
4. target_protein: Specific protein target if mentioned
5. confidence: 0.0-1.0 based on strength of evidence in the abstract

Return as JSON array. If no drug-innate immunity associations are found, return [].

Example:
[{{"drug_name": "Metformin", "pathway": "inflammasome", "effect": "inhibitor", "target_protein": "NLRP3", "confidence": 0.8}}]
"""

        try:
            response = await self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a biomedical NLP system that extracts structured drug-pathway associations from scientific abstracts. Be precise and conservative - only extract associations that are explicitly stated or strongly implied."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            # Handle both {"associations": [...]} and direct [...]
            if isinstance(data, dict):
                associations_data = data.get("associations", data.get("results", []))
            elif isinstance(data, list):
                associations_data = data
            else:
                return []

            associations = []
            for item in associations_data:
                associations.append(
                    DrugPathwayAssociation(
                        drug_name=item.get("drug_name", ""),
                        pathway=item.get("pathway", ""),
                        effect=item.get("effect", "unknown"),
                        target_protein=item.get("target_protein"),
                        confidence=float(item.get("confidence", 0.5)),
                        source_pmid=article.pmid,
                        source_title=article.title,
                        extraction_method="llm",
                    )
                )
            return associations

        except Exception as e:
            logger.error(f"LLM extraction failed for PMID {article.pmid}: {e}")
            return []

    async def scan_pathway(
        self,
        pathway_key: str,
        max_results: int = 100,
        min_date: str = "2020/01/01",
        extract: bool = True,
    ) -> dict:
        """Scan PubMed for drug repurposing opportunities in a specific innate immune pathway."""
        logger.info(f"Scanning pathway: {INNATE_IMMUNE_PATHWAYS[pathway_key]['name']}")
        query = self._build_repurposing_query(pathway_key)
        logger.info(f"Query: {query}")

        pmids = await self.search_pubmed(query, max_results=max_results, min_date=min_date)
        articles = await self.fetch_abstracts(pmids)

        result = {
            "pathway": pathway_key,
            "pathway_name": INNATE_IMMUNE_PATHWAYS[pathway_key]["name"],
            "query": query,
            "articles_found": len(articles),
            "articles": [a.model_dump() for a in articles],
            "associations": [],
            "scanned_at": datetime.utcnow().isoformat(),
        }

        if extract and self.llm_client:
            all_associations = []
            for article in articles:
                if article.abstract:
                    associations = await self.extract_associations_llm(article)
                    all_associations.extend(associations)
                    # Rate limit
                    await asyncio.sleep(0.5)
            result["associations"] = [
                {
                    "drug_name": a.drug_name,
                    "pathway": a.pathway,
                    "effect": a.effect,
                    "target_protein": a.target_protein,
                    "confidence": a.confidence,
                    "source_pmid": a.source_pmid,
                    "source_title": a.source_title,
                }
                for a in all_associations
            ]
            logger.info(f"Extracted {len(all_associations)} associations from {len(articles)} articles")

        return result

    async def scan_all_pathways(
        self, max_per_pathway: int = 50, min_date: str = "2020/01/01"
    ) -> dict:
        """Scan all innate immune pathways for drug repurposing candidates."""
        results = {}
        for pathway_key in INNATE_IMMUNE_PATHWAYS:
            results[pathway_key] = await self.scan_pathway(
                pathway_key, max_results=max_per_pathway, min_date=min_date
            )
            await asyncio.sleep(1)  # Be nice to NCBI servers
        return results


async def main():
    """Run the literature scanner as a standalone script."""
    import argparse

    parser = argparse.ArgumentParser(description="IMMUNEX Literature Scanner")
    parser.add_argument("--query", type=str, help="Custom PubMed query")
    parser.add_argument("--pathway", type=str, choices=list(INNATE_IMMUNE_PATHWAYS.keys()),
                        help="Specific pathway to scan")
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--min-date", type=str, default="2020/01/01")
    parser.add_argument("--output", type=str, default="results/literature_scan.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    async with LiteratureScanner() as scanner:
        if args.query:
            pmids = await scanner.search_pubmed(args.query, max_results=args.max_results)
            articles = await scanner.fetch_abstracts(pmids)
            result = {
                "query": args.query,
                "articles_found": len(articles),
                "articles": [a.model_dump() for a in articles],
            }
        elif args.pathway:
            result = await scanner.scan_pathway(
                args.pathway, max_results=args.max_results, min_date=args.min_date, extract=False
            )
        else:
            result = await scanner.scan_all_pathways(
                max_per_pathway=args.max_results, min_date=args.min_date
            )

        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

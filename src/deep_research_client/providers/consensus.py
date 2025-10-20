"""Consensus AI provider for academic research."""

from typing import List, Optional

import httpx

from . import ResearchProvider
from ..models import ResearchResult, ProviderConfig
from ..provider_params import ConsensusParams
from ..model_cards import ProviderModelCards, create_consensus_model_cards


class ConsensusProvider(ResearchProvider):
    """Provider for Consensus AI academic research API."""

    def __init__(self, config: ProviderConfig, params: Optional[ConsensusParams] = None):
        """Initialize Consensus provider."""
        self.params = params or ConsensusParams()
        super().__init__(config, self.params.model)

    def get_default_model(self) -> str:
        """Get default Consensus model."""
        return "Consensus Academic Search"

    @classmethod
    def model_cards(cls) -> ProviderModelCards:
        """Get model cards for Consensus provider."""
        return create_consensus_model_cards()

    async def research(self, query: str) -> ResearchResult:
        """Perform research using Consensus AI API."""
        if not self.is_available():
            raise ValueError(f"Consensus provider not available (API key: {bool(self.config.api_key)})")

        # Create HTTP client with timeout
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=self.config.timeout,
                write=30.0,
                pool=30.0,
            )
        )

        try:
            # Prepare headers for API authentication
            api_key = self.config.api_key
            if not api_key:
                raise ValueError("API key is required")

            headers = {
                'X-API-Key': api_key,
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'User-Agent': 'deep-research-client/1.0'
            }

            # Prepare query parameters
            params = {
                'query': query
            }

            # Make API request to Consensus
            response = await http_client.get(
                'https://api.consensus.app/v1/quick_search',
                headers=headers,
                params=params
            )

            response.raise_for_status()
            data = response.json()

            # Extract papers and format as markdown
            markdown_content = self._format_research_report(query, data)
            citations = self._extract_citations(data)

            return ResearchResult(
                markdown=markdown_content,
                citations=citations,
                provider=self.name,
                query=query
            )

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError("Invalid Consensus API key")
            elif e.response.status_code == 403:
                raise ValueError("Consensus API access denied - ensure you have applied for API access")
            elif e.response.status_code == 429:
                raise ValueError("Consensus API rate limit exceeded")
            else:
                raise ValueError(f"Consensus API error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Consensus API request failed: {e}")
        finally:
            await http_client.aclose()

    def _format_research_report(self, query: str, data: dict) -> str:
        """Format Consensus API response into a comprehensive research report."""
        report_lines = []

        # Introduction
        report_lines.append(f"# Academic Research Report: {query}")
        report_lines.append("")

        # Check if we have results
        results = data.get('results', [])
        if not results:
            report_lines.append("No academic papers found for this query.")
            return "\n".join(report_lines)

        report_lines.append(f"Found {len(results)} relevant academic papers from peer-reviewed sources.")
        report_lines.append("")

        # Summary section
        report_lines.append("## Research Summary")
        report_lines.append("")
        report_lines.append("Based on the analysis of peer-reviewed literature, here are the key findings:")
        report_lines.append("")

        # Group papers by relevance score if available
        high_relevance = []
        medium_relevance = []
        low_relevance = []

        for paper in results:
            relevance = paper.get('relevance_score', 0)
            if relevance >= 0.8:
                high_relevance.append(paper)
            elif relevance >= 0.6:
                medium_relevance.append(paper)
            else:
                low_relevance.append(paper)

        # High relevance papers (detailed analysis)
        if high_relevance:
            report_lines.append("### Key Research Findings")
            report_lines.append("")
            for i, paper in enumerate(high_relevance[:5], 1):  # Top 5
                title = paper.get('title', 'Untitled')
                authors = paper.get('authors', [])
                year = paper.get('year', 'Unknown year')
                journal = paper.get('journal', 'Unknown journal')
                citation_count = paper.get('citation_count', 0)

                author_names = [author.get('name', '') for author in authors[:3]]
                author_str = ', '.join(author_names)
                if len(authors) > 3:
                    author_str += ' et al.'

                report_lines.append(f"**{i}. {title}**")
                report_lines.append(f"- Authors: {author_str}")
                report_lines.append(f"- Published: {year} in {journal}")
                report_lines.append(f"- Citations: {citation_count}")

                # Add paper abstract or summary if available
                abstract = paper.get('abstract', '')
                if abstract:
                    # Truncate long abstracts
                    if len(abstract) > 300:
                        abstract = abstract[:300] + "..."
                    report_lines.append(f"- Summary: {abstract}")

                report_lines.append("")

        # Medium relevance papers (brief mentions)
        if medium_relevance:
            report_lines.append("### Additional Supporting Research")
            report_lines.append("")
            for paper in medium_relevance[:5]:  # Top 5
                title = paper.get('title', 'Untitled')
                authors = paper.get('authors', [])
                year = paper.get('year', 'Unknown year')

                author_names = [author.get('name', '') for author in authors[:2]]
                author_str = ', '.join(author_names)
                if len(authors) > 2:
                    author_str += ' et al.'

                report_lines.append(f"- **{title}** ({author_str}, {year})")

            report_lines.append("")

        # Research trends and patterns
        if results:
            report_lines.append("### Research Trends")
            report_lines.append("")

            # Analyze publication years
            years = [paper.get('year') for paper in results if paper.get('year')]
            if years:
                recent_papers = len([y for y in years if y and y >= 2020])
                total_papers = len(years)
                report_lines.append(f"- {recent_papers} of {total_papers} papers published since 2020, indicating {'active' if recent_papers/total_papers > 0.3 else 'moderate'} recent research activity")

            # Analyze journals
            journals = [paper.get('journal') for paper in results if paper.get('journal')]
            if journals:
                unique_journals = len(set(journals))
                report_lines.append(f"- Research published across {unique_journals} different journals, showing {'broad interdisciplinary' if unique_journals > 10 else 'focused disciplinary'} interest")

            # Citation analysis
            citations = [paper.get('citation_count', 0) for paper in results]
            if citations:
                high_impact = len([c for c in citations if c > 100])
                report_lines.append(f"- {high_impact} papers with >100 citations, indicating {'high' if high_impact > 5 else 'moderate'} research impact")

            report_lines.append("")

        # Methodology note
        report_lines.append("### Methodology")
        report_lines.append("")
        report_lines.append("This research report is based on academic papers indexed by Consensus AI, ")
        report_lines.append("which searches through peer-reviewed literature to provide evidence-based insights. ")
        report_lines.append("Papers are ranked by relevance to the query and include metadata such as ")
        report_lines.append("citation counts, publication details, and study characteristics.")

        return "\n".join(report_lines)

    def _extract_citations(self, data: dict) -> List[str]:
        """Extract citations from Consensus API response."""
        citations = []

        results = data.get('results', [])
        for paper in results:
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', [])
            year = paper.get('year', 'Unknown year')
            journal = paper.get('journal', 'Unknown journal')
            doi = paper.get('doi', '')
            url = paper.get('url', '')

            # Format author names
            author_names = [author.get('name', '') for author in authors[:5]]
            if len(authors) > 5:
                author_str = ', '.join(author_names) + ' et al.'
            else:
                author_str = ', '.join(author_names)

            # Create citation in APA-like format
            citation = f"{author_str} ({year}). {title}. {journal}."

            # Add DOI or URL if available
            if doi:
                citation += f" DOI: {doi}"
            elif url:
                citation += f" Retrieved from {url}"

            citations.append(citation)

        return citations
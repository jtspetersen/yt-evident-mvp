# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-02-05

### Fixed
- **Verification Accuracy**: Fixed asymmetric archetype gates - relaxed tier requirements for FALSE/LIKELY FALSE ratings while maintaining strict requirements for VERIFIED/LIKELY TRUE. FALSE ratings improved from 7% → 60% in test runs.
- **Source Quality**: Enhanced 6-tier source quality system now properly categorizes scholarly journals (tier 1), academic institutions (tier 2), government/international orgs (tier 3), research organizations (tier 4), news agencies (tier 5), and general websites (tier 6).
- **Claim Extraction**: Implemented overlapping chunks (5 segment overlap) with 85% similarity-based deduplication to prevent missing claims at chunk boundaries.
- **Empty Content Handling**: Added comprehensive diagnostic logging for when Ollama returns empty responses, capturing model parameters, prompt lengths, and response metadata.

### Added
- **Citation Enforcement**: Added explicit CRITICAL CITATION RULES to LLM prompts requiring all evidence snippets used in reasoning to be cited.
- **Rhetorical Analysis**: Added detection of when true facts are used to support false conclusions (false causation, cherry-picking, correlation as causation, appeal to fear, false dichotomies).
- **Source Filtering**: Expanded deny_domains to exclude reddit, forums, blogs, and social media platforms.
- **runvid Script**: New bash script that auto-activates virtual environment before running the pipeline.
- **Progress Bars**: Added tqdm progress indicators for LLM generation (streaming mode).

### Changed
- **README**: Comprehensive documentation update with CLI flags, configuration details, and usage examples.
- **Claim Prompts**: Enhanced extraction prompts to require COMPLETE and SELF-CONTAINED claims, with instructions to COMBINE related claims forming single logical arguments.
- **Requirements**: Added tqdm dependency for progress bars.

### Technical Details
**Files Modified**: 14 files (+892/-101 lines)
- Core: verify_claims.py, extract_claims.py, retrieve_evidence.py
- Tools: ollama_client.py
- Schema: verdict.py
- Config: config.yaml
- Docs: README.md
- New: runvid, runvid.bat

## [0.1.0] - 2026-02-04

### Added
- Initial MVP release
- Transcript normalization pipeline
- LLM-based claim extraction
- Web evidence retrieval via SearX
- Claim verification with LLM reasoning
- Scorecard generation (0-100 truthfulness score)
- Review video script generation
- Interactive claim review mode
- Configurable budgets and rate limits
- URL caching with TTL
- JSON-based data persistence

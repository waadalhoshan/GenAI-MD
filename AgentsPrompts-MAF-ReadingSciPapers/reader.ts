import { BaseAgent } from './base';
import { callLLM, FakeLLM } from '../lib/openai-client';
import { detectSections } from '../lib/pdf-parser';
import type { Message, Section, Entities, Citation } from '@shared/schema';

const READER_PROMPT = `You are a ReaderAgent specialized in analyzing scientific papers.

Your task is to extract entities from the provided paper sections. Focus on:
- Topics: Main research areas and subjects discussed
- Methods: Research methodologies, techniques, and approaches used
- Outcomes: Key results, findings, and conclusions
- Variables: Measured or studied variables

Return ONLY a valid JSON object with this structure:
{
  "entities": {
    "topics": ["topic1", "topic2", ...],
    "methods": ["method1", "method2", ...],
    "outcomes": ["outcome1", "outcome2", ...],
    "variables": ["var1", "var2", ...]
  }
}

Paper sections:
{sections}

Respond ONLY with valid JSON. No other text.`;

const CITATION_PROMPT = `You are a citation extraction expert. Extract all academic citations from the REFERENCES or BIBLIOGRAPHY section.

For each citation, extract:
- title: Paper/article title
- authors: Array of author names
- year: Publication year
- venue: Journal or conference name
- doi: DOI if present
- url: URL if present
- citationText: The original reference text exactly as it appears

Return ONLY a valid JSON object with this structure:
{
  "citations": [
    {
      "title": "Paper Title",
      "authors": ["Author One", "Author Two"],
      "year": "2023",
      "venue": "Journal Name",
      "doi": "10.1234/example",
      "citationText": "Original reference text..."
    }
  ]
}

References section:
{references}

Respond ONLY with valid JSON. No other text.`;

interface ReaderInput {
  text: string;
}

export class ReaderAgent extends BaseAgent {
  constructor(config: any) {
    super('ReaderAgent', config);
  }

  async run(input: ReaderInput): Promise<Message> {
    const startTime = Date.now();

    // Step 1: Detect sections
    const sections: Section[] = detectSections(input.text);

    // Step 2: Extract entities using LLM
    const sectionsText = sections.map(s => `${s.name}:\n${s.text.substring(0, 500)}`).join('\n\n');
    const prompt = READER_PROMPT.replace('{sections}', sectionsText);

    let response: string;
    if (this.config.provider === 'fake') {
      response = await FakeLLM.call(prompt);
    } else {
      response = await callLLM(prompt, {
        model: this.config.model,
        temperature: this.config.temperature,
        maxTokens: this.config.maxTokens,
        seed: this.config.seed,
        responseFormat: 'json',
      });
    }

    const parsed = this.parseJSONResponse(response);
    const entities: Entities = parsed.entities || parsed;

    // Step 3: Extract citations from References section
    let citations: Citation[] = [];
    const referencesSection = sections.find(s => 
      s.name.toLowerCase().includes('reference') || 
      s.name.toLowerCase().includes('bibliography')
    );

    if (referencesSection) {
      const citationPrompt = CITATION_PROMPT.replace('{references}', referencesSection.text);
      
      let citationResponse: string;
      if (this.config.provider === 'fake') {
        citationResponse = await FakeLLM.call(citationPrompt);
      } else {
        citationResponse = await callLLM(citationPrompt, {
          model: this.config.model,
          temperature: 0.3, // Lower temperature for better extraction accuracy
          maxTokens: this.config.maxTokens,
          seed: this.config.seed,
          responseFormat: 'json',
        });
      }

      const citationParsed = this.parseJSONResponse(citationResponse);
      citations = citationParsed.citations || [];
    }

    const elapsed = Date.now() - startTime;

    return this.createMessage(
      `Detected ${sections.length} sections, extracted entities, and found ${citations.length} citations`,
      {
        kind: 'sections',
        payload: { sections, entities, citations },
      },
      { elapsedMs: elapsed }
    );
  }
}

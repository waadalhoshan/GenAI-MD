import { BaseAgent } from './base';
import { callLLM, FakeLLM } from '../lib/openai-client';
import type { Message, Section, Entities, Analysis } from '@shared/schema';

const ANALYZER_PROMPT = `You are an AnalyzerAgent specialized in critically analyzing scientific papers.

Given the paper sections and extracted entities, provide a comprehensive analysis.

Sections:
{sections}

Entities:
{entities}

Return ONLY a valid JSON object with this structure:
{
  "researchQuestion": "What is the main research question?",
  "methodology": "Description of the research methodology",
  "dataset": "Description of dataset/sample (if applicable)",
  "keyFindings": ["finding1", "finding2", ...],
  "metrics": ["metric1", "metric2", ...],
  "assumptions": ["assumption1", "assumption2", ...],
  "limitations": ["limitation1", "limitation2", ...],
  "threatsToValidity": ["threat1", "threat2", ...],
  "evidenceStrength": "weak" | "moderate" | "strong"
}

Be concise, neutral, and avoid verbatim copying. Respond ONLY with valid JSON.`;

interface AnalyzerInput {
  sections: Section[];
  entities: Entities;
}

export class AnalyzerAgent extends BaseAgent {
  constructor(config: any) {
    super('AnalyzerAgent', config);
  }

  async run(input: AnalyzerInput): Promise<Message> {
    const startTime = Date.now();

    const sectionsText = input.sections.map(s => `${s.name}:\n${s.text.substring(0, 400)}`).join('\n\n');
    const entitiesText = JSON.stringify(input.entities, null, 2);

    const prompt = ANALYZER_PROMPT
      .replace('{sections}', sectionsText)
      .replace('{entities}', entitiesText);

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

    const analysis: Analysis = this.parseJSONResponse(response);
    const elapsed = Date.now() - startTime;

    return this.createMessage(
      `Analysis complete: ${analysis.keyFindings.length} key findings identified`,
      {
        kind: 'analysis',
        payload: analysis,
      },
      { elapsedMs: elapsed }
    );
  }
}

import { BaseAgent } from './base';
import { callLLM, FakeLLM } from '../lib/openai-client';
import type { Message, Analysis, Insights } from '@shared/schema';

const INSIGHT_PROMPT = `You are an InsightAgent specialized in creating educational content from scientific research.

Given this analysis of a research paper, generate educational insights for non-experts.

Analysis:
{analysis}

Return ONLY a valid JSON object with this structure:
{
  "explain": "A short, non-technical conceptual explanation (2-3 sentences)",
  "questions": ["question1", "question2", ...],
  "apply": ["application1", "application2", ...]
}

Requirements:
- Explain: Make it accessible to non-experts, use analogies if helpful
- Questions: 3-7 reflection/learning questions that encourage critical thinking
- Apply: 3-7 practical ways a learner could use or test these insights

Respond ONLY with valid JSON.`;

interface InsightInput {
  analysis: Analysis;
}

export class InsightAgent extends BaseAgent {
  constructor(config: any) {
    super('InsightAgent', config);
  }

  async run(input: InsightInput): Promise<Message> {
    const startTime = Date.now();

    const analysisText = JSON.stringify(input.analysis, null, 2);
    const prompt = INSIGHT_PROMPT.replace('{analysis}', analysisText);

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

    const insights: Insights = this.parseJSONResponse(response);
    const elapsed = Date.now() - startTime;

    return this.createMessage(
      `Generated ${insights.questions.length} questions and ${insights.apply.length} applications`,
      {
        kind: 'insights',
        payload: insights,
      },
      { elapsedMs: elapsed }
    );
  }
}

import { BaseAgent } from './base';
import { callLLM, FakeLLM } from '../lib/openai-client';
import type { Message, Analysis, Insights, Evaluation } from '@shared/schema';

const EVALUATOR_PROMPT = `You are an EvaluatorAgent specialized in reviewing analysis quality.

Review the following analysis and insights for coherence, consistency, and educational value.

Analysis:
{analysis}

Insights:
{insights}

Return ONLY a valid JSON object with this structure:
{
  "coherenceScore": 0.0-1.0,
  "consistencyScore": 0.0-1.0,
  "educationalValueScore": 0.0-1.0,
  "rankedPoints": [
    {"rank": 1, "content": "most valuable point", "importance": 0.0-1.0},
    ...
  ],
  "suggestedEdits": [
    {"section": "section name", "original": "text", "suggested": "improved text", "reason": "why"},
    ...
  ]
}

Evaluate:
- Coherence: How well do ideas connect and flow?
- Consistency: Are there contradictions or inconsistencies?
- Educational Value: How useful is this for learning?

Rank the 3-5 most valuable points and optionally suggest improvements.
Respond ONLY with valid JSON.`;

interface EvaluatorInput {
  analysis: Analysis;
  insights: Insights;
}

export class EvaluatorAgent extends BaseAgent {
  constructor(config: any) {
    super('EvaluatorAgent', config);
  }

  async run(input: EvaluatorInput): Promise<Message> {
    const startTime = Date.now();

    const analysisText = JSON.stringify(input.analysis, null, 2);
    const insightsText = JSON.stringify(input.insights, null, 2);

    const prompt = EVALUATOR_PROMPT
      .replace('{analysis}', analysisText)
      .replace('{insights}', insightsText);

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

    const evaluation: Evaluation = this.parseJSONResponse(response);
    const elapsed = Date.now() - startTime;

    return this.createMessage(
      `Evaluation complete: coherence ${(evaluation.coherenceScore * 100).toFixed(0)}%`,
      {
        kind: 'evaluation',
        payload: evaluation,
      },
      { elapsedMs: elapsed }
    );
  }
}

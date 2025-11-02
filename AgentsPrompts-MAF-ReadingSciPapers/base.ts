import type { Message, Artifact } from '@shared/schema';

export interface AgentConfig {
  model: string;
  temperature: number;
  maxTokens: number;
  seed?: number;
  provider: 'openai' | 'fake';
}

export abstract class BaseAgent {
  protected name: string;
  protected config: AgentConfig;

  constructor(name: string, config: AgentConfig) {
    this.name = name;
    this.config = config;
  }

  abstract run(input: any): Promise<Message>;

  protected createMessage(
    content: string,
    artifact?: Artifact,
    metadata?: Record<string, any>
  ): Message {
    return {
      id: `${this.name}-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
      timestamp: new Date().toISOString(),
      sender: this.name,
      role: 'agent',
      content,
      artifact,
      metadata,
    };
  }

  protected parseJSONResponse(response: string): any {
    try {
      return JSON.parse(response);
    } catch (error) {
      // Try to extract JSON from markdown code blocks
      const jsonMatch = response.match(/```(?:json)?\s*(\{[\s\S]*\})\s*```/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[1]);
      }
      throw new Error('Failed to parse JSON response from LLM');
    }
  }
}

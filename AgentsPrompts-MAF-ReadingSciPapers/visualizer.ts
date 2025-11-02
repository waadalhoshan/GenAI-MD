import { BaseAgent } from './base';
import type { Message, PipelineRun } from '@shared/schema';

export class VisualizerAgent extends BaseAgent {
  constructor(config: any) {
    super('VisualizerAgent', config);
  }

  async run(run: PipelineRun): Promise<Message> {
    const startTime = Date.now();

    // Visualizer primarily formats data for display
    // In this implementation, formatting is handled client-side
    // This agent serves as a validation/completion step

    const elapsed = Date.now() - startTime;

    return this.createMessage(
      'Visualization formatting complete',
      {
        kind: 'visualization',
        payload: { formatted: true },
      },
      { elapsedMs: elapsed }
    );
  }
}

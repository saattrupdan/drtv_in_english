export interface Cue {
  start: number;
  end: number;
  text: string;
}

export type SubMode = "off" | "dansk" | "english";

export type PortMessage =
  | { type: "episode-active"; episodeId: string; url: string }
  | { type: "request-translate"; episodeId: string; playhead?: number }
  | { type: "seek"; time: number }
  | { type: "cancel" };

export type PortEvent =
  | { type: "status"; state: TranslationState; detail?: string }
  | { type: "schedule"; starts: number[] }
  | { type: "cues"; cues: Cue[] }
  | { type: "done"; total: number }
  | { type: "error"; message: string };

export type TranslationState =
  | "idle"
  | "waiting-for-vtt"
  | "fetching-vtt"
  | "parsing"
  | "translating"
  | "done";

export const PORT_NAME = "drtv-en";

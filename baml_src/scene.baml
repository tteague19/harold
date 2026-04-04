enum EmotionalTone {
  NEUTRAL
  JOYFUL
  ANXIOUS
  ANGRY
  CONFUSED
  EXCITED
  MELANCHOLY
  PLAYFUL
  SINCERE
  ABSURD
}

class SceneResponse {
  dialogue string @description("The character's dialogue or narration")
  stage_direction string? @description("Optional physical action or stage direction")
  emotional_tone EmotionalTone @description("The emotional quality of this beat")
  callback_used string? @description("Reference to an earlier scene element")
}

class SceneSummaryOutput {
  summary string @description("Brief narrative summary of the scene")
  key_moments string[] @description("Notable beats or discoveries in the scene")
  techniques_used string[] @description("Improv techniques observed during the scene")
}

function SummarizeScene(transcript: string, setting: string, suggestion: string) -> SceneSummaryOutput {
  client Anthropic
  prompt #"
    You are an expert improv comedy analyst trained in UCB (Upright Citizens Brigade)
    methodology. Analyze the following scene transcript and produce a structured summary.

    Setting: {{ setting }}
    Audience suggestion: {{ suggestion }}

    Transcript:
    {{ transcript }}

    Identify the key moments, the game of the scene if one emerged, and which
    improv techniques were used (e.g. yes-and, heightening, callbacks, game of
    the scene, group mind, tag-out, walk-on).

    {{ ctx.output_format }}
  "#
}

test SummarizeSceneBasic {
  functions [SummarizeScene]
  args {
    transcript "A: Welcome to the moon bakery! B: Thanks, I can't believe we finally opened. A: The croissants keep floating away though. B: That's why we use the gravity sprinkles. A: Right, the gravity sprinkles! Last week we ran out and lost an entire batch of baguettes to orbit."
    setting "A bakery on the moon"
    suggestion "bakery"
  }
}

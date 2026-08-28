#include "thinking_control.h"

#include <cctype>
#include <cstdio>
#include <exception>
#include <string>

namespace bmoe {

const char * think_control_name(ThinkControl c) {
    switch (c) {
    case ThinkControl::Template:
        return "template";
    case ThinkControl::Prefill:
        return "prefill";
    case ThinkControl::None:
        return "none";
    }
    return "template";
}

} // namespace bmoe

namespace bmoe::detail {

namespace {

// What a natively-supporting template puts inside the closed span. See add_no_think_prefill:
// whitespace, so the engine contributes no words to the model's own reasoning.
const char * const kEmptyReasoning = "\n\n";

// Render a one-turn conversation the way generate() renders a real one — same jinja path, same
// reasoning format — so what the probe observes is what generation will produce. The message text
// is irrelevant: templates branch on the flag, never on the words.
common_chat_params apply_probe(const common_chat_templates * tmpls, bool enable_thinking, bool prefill) {
    common_chat_msg user;
    user.role = "user";
    user.content = "hi";

    common_chat_templates_inputs inputs;
    inputs.messages = {user};
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = enable_thinking;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    if (prefill) add_no_think_prefill(inputs);

    return common_chat_templates_apply(const_cast<common_chat_templates *>(tmpls), inputs);
}

} // namespace

void add_no_think_prefill(common_chat_templates_inputs & inputs) {
    common_chat_msg prefill;
    prefill.role = "assistant";
    prefill.reasoning_content = kEmptyReasoning;
    // content stays empty: the handler renders the closing tag and then nothing, leaving the model
    // at the first token of its answer. Nothing is appended that would have to be stripped back off.
    inputs.messages.push_back(prefill);
    inputs.continue_final_message = COMMON_CHAT_CONTINUATION_CONTENT;
}

ThinkControl probe_think_control(const common_chat_templates * tmpls) {
    if (tmpls == nullptr) return ThinkControl::Template;
    try {
        const common_chat_params off_p = apply_probe(tmpls, /*enable_thinking=*/false, /*prefill=*/false);
        const std::string & off = off_p.prompt;
        const std::string on = apply_probe(tmpls, /*enable_thinking=*/true, /*prefill=*/false).prompt;
        if (on != off) return ThinkControl::Template;

        // Does the model own a reasoning span — a `<think>`-style pair it opens and closes itself?
        //
        // Both halves are needed. A declared tag alone proves nothing, because handlers publish the
        // pair for a whole family: the non-reasoning members (LFM2-8B-A1B, LFM2.5-Instruct) advertise
        // `<think>` they never emit. Requiring the template to actually USE it is the same test
        // llama.cpp applies before wiring up reasoning extraction for that family.
        const std::string src = common_chat_templates_source(tmpls);
        const bool declares_span =
            !off_p.thinking_start_tag.empty() && src.find(off_p.thinking_start_tag) != std::string::npos;

        // Nothing indicates this model reasons at all: a tag its family publishes but this template
        // never emits. There is nothing to suppress, so pass the flag and add nothing — claiming it
        // "always reasons" would be a louder lie than the silence this probe exists to end.
        if (!declares_span) return ThinkControl::Template;

        const std::string prefilled = apply_probe(tmpls, /*enable_thinking=*/false, /*prefill=*/true).prompt;

        // The prefill changes nothing, so there is no lever at all.
        if (prefilled == off) return ThinkControl::None;

        // Where did the prefill LEAVE the model? Asking that, rather than whether a start tag was
        // published, is what tells the two regimes apart — and it stays true across a submodule bump,
        // because it reads the rendered prompt instead of a field whose population is upstream's
        // choice. Harmony used to publish no start tag and now publishes one; nothing about either
        // model changed.
        //
        //   ends ON the closing tag  — the span is merely CLOSED, and closing it is a suggestion to
        //                              a model that opens its own: LFM2.5 reasons straight past a
        //                              pre-closed empty span and emits it untagged into the answer
        //                              (issue #82), worse than leaving it alone.
        //   ends PAST it             — the prompt has moved into a later section the format itself
        //                              separates, so the skipped one is not something the model can
        //                              decline (harmony/gpt-oss ends on `<|channel|>final<|message|>`).
        auto ends_with = [](const std::string & s, const std::string & suffix) {
            return !suffix.empty() && s.size() >= suffix.size() &&
                   s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
        };
        std::string tail = prefilled;
        while (!tail.empty() && std::isspace((unsigned char) tail.back())) tail.pop_back();
        for (const std::string & end_tag : off_p.thinking_end_tags) {
            if (ends_with(tail, end_tag)) return ThinkControl::None;
        }
        return ThinkControl::Prefill;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "bmoe: thinking-control probe failed (%s); assuming the template honours it\n", e.what());
        return ThinkControl::Template;
    }
}

} // namespace bmoe::detail

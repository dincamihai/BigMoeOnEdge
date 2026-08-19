#include "canonical_prefix.h"

namespace bmoe::detail {

std::string canonical_prefix_text(const struct common_chat_templates * tmpls,
                                  const std::vector<common_chat_msg> & history,
                                  const std::vector<common_chat_tool> & tools,
                                  bool think) {
    // A past assistant turn keeps no reasoning in any template that distinguishes them, and this
    // turn's answer is about to become a past one.
    std::vector<common_chat_msg> clean_history = history;
    for (common_chat_msg & m : clean_history) m.reasoning_content.clear();

    // The sentinel's ROLE is fixed and only its CONTENT varies. Varying the role instead moves
    // whatever the template keys reasoning retention on — DeepSeek-V4 tracks the index of the last
    // user message — so the two renders disagree at the current answer's own header and the answer
    // is cut out of the canonical sequence entirely.
    auto render_with = [&](const char * content) {
        std::vector<common_chat_msg> h = clean_history;
        common_chat_msg sentinel;
        sentinel.role = "user";
        sentinel.content = content;
        h.push_back(sentinel);

        common_chat_templates_inputs inputs;
        inputs.messages = h;
        inputs.tools = tools;
        inputs.add_generation_prompt = false;
        inputs.use_jinja = true;
        inputs.enable_thinking = think;
        inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
        return common_chat_templates_apply(const_cast<common_chat_templates *>(tmpls), inputs).prompt;
    };

    // Two contents with no common first character, so the renders can agree no further than the
    // point where the sentinel's own text begins.
    const std::string a = render_with("x");
    const std::string b = render_with("y");
    size_t agree = 0;
    while (agree < a.size() && agree < b.size() && a[agree] == b[agree]) ++agree;
    return a.substr(0, agree);
}

} // namespace bmoe::detail

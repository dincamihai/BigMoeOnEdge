// Canonical-prefix derivation, against real chat templates with no model.
//
// The bug this pins: the derivation rendered the history twice with a throwaway message that
// differed in ROLE — user in one render, assistant in the other — and kept the tokens the two
// agreed on. That assumes the only thing a role change moves is where the two renders diverge.
// It is not. DeepSeek-V4's template records the index of the LAST USER message and keeps an
// assistant turn's reasoning only when that turn comes after it:
//
//     {%- set keep_reasoning = tp.has or (loop.index0 > last_user_idx.value) -%}
//
// A user sentinel puts the answer BEFORE the last user message (reasoning stripped), an assistant
// sentinel leaves it after (reasoning kept). The renders therefore disagree at the answer's own
// header and the agreed prefix stops at the end of the user turn — the answer never reaches the
// canonical sequence at all, and every following turn re-prefills it. Measured on evo before the
// fix: an 889-token prompt reused 29 tokens.
//
// The fix is to hold the ROLE fixed and vary only the sentinel's CONTENT, so anything the template
// keys on is identical in both renders and they can only diverge inside the sentinel itself.
//
// Assertions are explicit (not <cassert>) because the Release gates build with NDEBUG, which
// would compile assert() out.

#include "canonical_prefix.h"

#include "chat.h"
#include "common.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#if !defined(BMOE_TMPL_DEEPSEEK_V4) || !defined(BMOE_TMPL_GLM46) || !defined(BMOE_TMPL_QWEN3)
#error "template paths must be defined by the build"
#endif

static int failures = 0;

static void expect(const char * name, bool ok, const std::string & detail = {}) {
    if (ok) {
        std::printf("[PASS] %s\n", name);
    } else {
        std::printf("[FAIL] %s%s%s\n", name, detail.empty() ? "" : "\n  ", detail.c_str());
        ++failures;
    }
}

static std::string read_file(const char * path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::printf("[FAIL] cannot open template: %s\n", path);
        ++failures;
        return {};
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static common_chat_templates_ptr load(const char * path) {
    return common_chat_templates_init(/*model*/ nullptr, read_file(path));
}

// A finished turn: the conversation the canonical sequence has to describe.
static std::vector<common_chat_msg> history() {
    common_chat_msg sys;
    sys.role = "system";
    sys.content = "Be terse.";
    common_chat_msg user;
    user.role = "user";
    user.content = "Reply with the magic word.";
    common_chat_msg asst;
    asst.role = "assistant";
    // Deliberately a string that appears NOWHERE in the prompt: an answer echoing the question
    // would let a derivation that stops BEFORE the answer still satisfy the assertion below.
    asst.content = "xyzzy";  // reasoning_content deliberately empty, as the engine commits it
    return {sys, user, asst};
}

// What the NEXT turn actually sends: the same history plus a new user turn, with a generation
// prompt. This is the string the canonical sequence is required to be a prefix of — the whole
// contract, stated as the test's oracle rather than as an expectation about markers.
static std::string next_prompt(const common_chat_templates * tmpls, bool think) {
    std::vector<common_chat_msg> h = history();
    common_chat_msg nxt;
    nxt.role = "user";
    nxt.content = "And again.";
    h.push_back(nxt);

    common_chat_templates_inputs inputs;
    inputs.messages = h;
    inputs.add_generation_prompt = true;
    inputs.use_jinja = true;
    inputs.enable_thinking = think;
    inputs.reasoning_format = COMMON_REASONING_FORMAT_AUTO;
    return common_chat_templates_apply(const_cast<common_chat_templates *>(tmpls), inputs).prompt;
}

static void check(const char * name, const char * path, bool think) {
    common_chat_templates_ptr tmpls = load(path);
    if (!tmpls) {
        expect(name, false, "template did not load");
        return;
    }
    const std::string canon = bmoe::detail::canonical_prefix_text(tmpls.get(), history(), {}, think);
    const std::string later = next_prompt(tmpls.get(), think);

    // 1. The contract: canonical must be a prefix of what the next turn sends, or the next turn
    //    cannot extend it and pays a full prefill.
    expect((std::string(name) + ": is a prefix of the next prompt").c_str(),
           !canon.empty() && later.rfind(canon, 0) == 0,
           "canonical:\n  " + canon + "\nnext prompt:\n  " + later);

    // 2. The regression itself: the answer has to be IN it. Without this the first assertion
    //    passes on the broken derivation — a prefix that stops before the answer is still a
    //    prefix, which is exactly why the defect survived the original 3-turn validation.
    expect((std::string(name) + ": reaches the answer").c_str(),
           canon.find("xyzzy") != std::string::npos,
           "answer missing from canonical:\n  " + canon);
}

int main() {
    try {
        // DeepSeek-V4: the template whose last-user-index rule defeated the role-varying pair.
        check("deepseek-v4 think-on", BMOE_TMPL_DEEPSEEK_V4, /*think*/ true);
        check("deepseek-v4 think-off", BMOE_TMPL_DEEPSEEK_V4, /*think*/ false);
        // GLM-4.6: same family of rule, and the worst measured case — the old derivation cut at
        // 22 characters, still inside the preamble, so canonical held nothing at all.
        check("glm-4.6 think-on", BMOE_TMPL_GLM46, /*think*/ true);
        // Qwen3: the model that already worked. This is the no-regression half.
        check("qwen3 think-on", BMOE_TMPL_QWEN3, /*think*/ true);
    } catch (const std::exception & e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++failures;
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all canonical-prefix checks passed\n");
    return 0;
}

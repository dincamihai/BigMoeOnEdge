#pragma once

// Deriving the CANONICAL text of a conversation: "this conversation as a LATER prompt will
// contain it". Kept separate from session.cpp so the derivation is unit-testable against real
// chat templates without a model or a live session — the seam that hid the defect below.
//
// Internal header: includes llama.cpp's `common` (not the stable public API), so it must not
// be pulled into core/include/bmoe/. See docs/seam.md.

#include "chat.h"

#include <string>
#include <vector>

namespace bmoe::detail {

// The tokens the canonical sequence must hold are the ones a LATER prompt opens with, and no
// single render produces them: a template renders the assistant turn that is CURRENT differently
// from the same turn once it is past.
//
// So do not model the template. Render the same history twice with a throwaway message appended
// and keep what the two renders agree on. The sentinels must differ ONLY in their content — a
// pair that differs in ROLE also moves whatever the template keys reasoning retention on
// (DeepSeek-V4 tracks the index of the last user message), which makes the two renders disagree
// at the current answer's header and cuts the answer off entirely.
//
// The agreed prefix runs through the answer and ends after the next turn's role header, which is
// a valid prefix of the next prompt whenever that turn is a user turn. When it is not, the caller
// finds canonical is not an extension and lays it down again from zero.
std::string canonical_prefix_text(const struct common_chat_templates * tmpls,
                                  const std::vector<common_chat_msg> & history,
                                  const std::vector<common_chat_tool> & tools,
                                  bool think);

} // namespace bmoe::detail

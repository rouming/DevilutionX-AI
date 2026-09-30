#pragma once

#include "utils/attributes.h"

namespace devilution {

/**
 * @brief Don't load UI or show Messageboxes or other user-interaction. Needed for unit tests.
 */
extern DVL_API_FOR_TEST bool HeadlessMode;

/**
 * @brief Suppress rendering and audio. Set via RING_ENTRY_KEY_FAST_FORWARD for
 * replay fast-forward: the window stays open but nothing is drawn or played.
 */
extern bool FastForwardMode;

} // namespace devilution

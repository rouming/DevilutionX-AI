#pragma once

#include "utils/ring.h"
#include "monster.h"
#include "player.h"

namespace devilution {

struct MonsterTypeInfo {
	uint8_t level;
	uint8_t walk_frames;   // frames[MonsterGraphic::Walk]
	uint8_t attack_frames; // frames[MonsterGraphic::Attack]
};

namespace shared {
	extern struct ring_queue   input_queue;
	extern struct ring_queue   events_queue;
	extern uint64_t            game_ticks;
	extern uint64_t            game_saves;
	extern uint64_t            game_loads;
	extern struct Player       player;
	extern MonsterTypeInfo     monster_type_info[MaxLvlMTypes];

	// Global maxes computed from game data files at startup.
	// Used by the AI agent to normalize observation channels globally.
	extern uint8_t             max_monster_level;  // monstdat.tsv, normalized threat_level
	extern uint8_t             max_walk_frames;    // monstdat.tsv, normalized move_speed
	extern uint8_t             max_attack_frames;  // monstdat.tsv, normalized attack_speed
	extern uint8_t             max_weapon_dam;     // itemdat.tsv, normalized _pIMaxDam

	// Class attribute caps for the current player's class.
	extern ClassAttributes     player_class_attrs;

	void share_diablo_state(const std::string &path);
}
}

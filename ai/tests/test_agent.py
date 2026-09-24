#!/usr/bin/env python3
"""test_agent.py - Integration tests for AgentAI inventory management.

Starts its own headless Diablo instance (reads diablo-ai.ini for paths),
runs the tests, then shuts the engine down.

Usage:
    cd ai
    python tests/test_agent.py
"""

import sys
import os
import configparser
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# dx, ring, diablo_state, AgentAI are imported after the generator runs.
dx = ring = diablo_state = AgentAI = None

# Cursor values from Source/itemdat.h (not exported via dx)
_ICURS_LONG_SWORD         = 60
_ICURS_BUCKLER            = 83
_ICURS_TWO_HANDED_SWORD   = 110
_ICURS_POTION_OF_HEALING  = 32
_ICURS_SCROLL_OF          = 1
_ICURS_RING               = 12
_ICURS_AMULET             = 45

# item_index values counted from IDI_GOLD=0
_IDI_HEAL     = 24
_IDI_IDENTIFY = 26

# Settle after at most this many agent ticks; enough for equip + backup drop.
_SETTLE_MAX_TICKS = 15


class _FullState:
    """Full snapshot of player body + inventory + belt at one point in time."""

    def __init__(self, player):
        none_type  = dx.ItemType.None_.value
        inv_first  = dx.inv_item.INVITEM_INV_FIRST.value

        # body: slot index -> seed (only non-empty slots)
        self.body = {}
        for i in range(inv_first):
            item = player.InvBody[i]
            if int(item._itype) != none_type:
                self.body[i] = int(item._iSeed)

        # all seeds in inventory and belt
        inv_seeds = set()
        for i in range(int(player._pNumInv)):
            item = player.InvList[i]
            if int(item._itype) != none_type:
                inv_seeds.add(int(item._iSeed))
        for item in player.SpdList:
            if int(item._itype) != none_type:
                inv_seeds.add(int(item._iSeed))

        self.seeds = set(self.body.values()) | inv_seeds

    def assert_body(self, slot, seed, label=""):
        actual = self.body.get(slot, 0)
        suffix = f" ({label})" if label else ""
        assert actual == seed, \
            f"body[{slot}]={actual:#010x} expected {seed:#010x}{suffix}"

    def assert_absent(self, seed, label=""):
        suffix = f" ({label})" if label else ""
        assert seed not in self.seeds, \
            f"seed {seed:#010x} should be absent but is present{suffix}"

    def assert_present(self, seed, label=""):
        suffix = f" ({label})" if label else ""
        assert seed in self.seeds, \
            f"seed {seed:#010x} should be present but is absent{suffix}"

def _reset_game(game):
    # Reset to dungeon level 1 for the 2H backup scenario.
    RE = ring.RingEntryType
    game.submit_key(RE.RING_ENTRY_KEY_NEW | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
                    data=((1 << 1) | 1, 0))

def _gift(game, seed, itype, iloc, iclass, icurs, name, mindam, maxdam, ac=0,
          minstr=0, mindex=0, imiscid=0, ididx=0, identified=1, imagical=0,
          pldam_mod=0, plhp=0, ispell=0, durability=255, maxdur=255):
    """Fill GiftItem and trigger INV_GIFT_ITEM ring command."""
    g = game.state.GiftItem
    # Explicit field assignment avoids slice issues on nested struct fields
    # (position, AnimInfo) that lack sub-array dtype.
    g['_iSeed']       = seed
    g['_iCreateInfo'] = 0
    g['_itype']       = itype
    g['_iAnimFlag']   = 0
    g['_iDelFlag']    = 0
    g['_iIdentified'] = identified
    g['_iMagical']    = imagical
    g['_iLoc']        = iloc
    g['_iClass']      = iclass
    g['_iCurs']       = icurs
    g['_ivalue']      = 1000
    g['_iIvalue']     = 1000
    g['_iMinDam']     = mindam
    g['_iMaxDam']     = maxdam
    g['_iAC']         = ac
    g['_iFlags']      = 0
    g['_iMiscId']     = imiscid
    g['_iSpell']      = ispell
    g['IDidx']        = ididx
    g['_iCharges']    = 0
    g['_iMaxCharges'] = 0
    g['_iDurability'] = durability
    g['_iMaxDur']     = maxdur
    g['_iPLDam']      = 0
    g['_iPLToHit']    = 0
    g['_iPLAC']       = 0
    g['_iPLStr']      = 0
    g['_iPLMag']      = 0
    g['_iPLDex']      = 0
    g['_iPLVit']      = 0
    g['_iPLFR']       = 0
    g['_iPLLR']       = 0
    g['_iPLMR']       = 0
    g['_iPLMana']     = 0
    g['_iPLDamMod']   = pldam_mod
    g['_iPLHP']       = plhp
    g['_iPLGetHit']   = 0
    g['_iPLLight']    = 0
    g['_iSplLvlAdd']  = 0
    g['_iFMinDam']    = 0
    g['_iFMaxDam']    = 0
    g['_iLMinDam']    = 0
    g['_iLMaxDam']    = 0
    g['_iPLEnAc']     = 0
    g['_iPrePower']   = 0
    g['_iSufPower']   = 0
    g['_iMinStr']     = minstr
    g['_iMinMag']     = 0
    g['_iMinDex']     = mindex
    g['_iStatFlag']   = 1
    b = name.encode('ascii')[:63]
    g['_iName'][:len(b)]  = np.frombuffer(b, dtype=np.int8)
    g['_iName'][len(b):]  = 0
    g['_iIName'][:len(b)] = np.frombuffer(b, dtype=np.int8)
    g['_iIName'][len(b):] = 0
    RE = ring.RingEntryType
    game.submit_key(RE.RING_ENTRY_KEY_INV_GIFT_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS)


def _settle(agent, max_ticks=_SETTLE_MAX_TICKS):
    """Run agent ticks until quiescent (queue empty, no pending equips, no new items)."""
    for _ in range(max_ticks):
        agent._tick()
        if (not agent._action_queue and
                not agent._pending_equip and
                not agent._inv_changed):
            return
    raise AssertionError(
        f"agent did not settle after {max_ticks} ticks; "
        f"queue={agent._action_queue}, pending={agent._pending_equip}")


class _StandModel:
    """Dummy model that always returns Stand so the hero stays put in dungeon."""
    _STAND = 8  # ActionEnum.Stand.value (Walk_N=0 .. Walk_NW=7, Stand=8)
    acmodel = type('acmodel', (), {'has_automap': False})()

    def reset(self): pass
    def act(self, obs): return self._STAND


class InvTests:
    """Sequential inventory management tests against a live engine instance."""

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._s0 = None  # full state after test_01 (reference baseline)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _settle(self):
        _settle(self.agent)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_sword(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def _gift_shield(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac)

    def _gift_2h_sword(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def _gift_potion(self, seed, name):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_POTION_OF_HEALING,
              name=name, mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_HEAL.value,
              ididx=_IDI_HEAL)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_01_initial_state(self):
        """Warrior starts with a weapon in HAND_LEFT and a shield in HAND_RIGHT.
        Records the full baseline state for subsequent tests."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        self._s0 = self._state()
        assert int(self.game.safe_state.player.InvBody[HL]._iClass) == \
            dx.item_class.ICLASS_WEAPON.value, \
            f"HAND_LEFT is not a weapon: class={self._s0.body.get(HL)}"
        assert int(self.game.safe_state.player.InvBody[HR]._itype) == \
            dx.ItemType.Shield.value, \
            f"HAND_RIGHT is not a shield"
        print(f"  sword  seed={self._s0.body[HL]:#010x}")
        print(f"  shield seed={self._s0.body[HR]:#010x}")
        print(f"  all    seeds={sorted(f'{s:#010x}' for s in self._s0.seeds)}")

    def test_02_gift_better_sword_equipped(self):
        """Gifting a stronger 1H sword equips it and destroys the old one."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        old_sword  = self._s0.body[HL]
        old_shield = self._s0.body[HR]

        self._gift_sword(seed=0xDEAD0001, name="Test Sword", mindam=10, maxdam=20)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0001, "new sword equipped")
        s.assert_body(HR, old_shield,  "shield unchanged")
        s.assert_absent(old_sword,     "old sword destroyed")

    def test_03_gift_shield_equipped(self):
        """Gifting a better shield equips it and destroys the old one."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        old_shield = self._s0.body[HR]

        # ac=20 keeps eff(1H+sh) below eff(2H_avg100)=742 for tests 04-10.
        self._gift_shield(seed=0xDEAD0002, name="Test Shield", ac=20)
        self._settle()
        s = self._state()

        s.assert_body(HR, 0xDEAD0002, "new shield equipped")
        s.assert_body(HL, 0xDEAD0001, "sword from test_02 still equipped")
        s.assert_absent(old_shield,    "old shield destroyed")

    def test_04_weaker_2h_rejected(self):
        """A 2H whose eff < current 1H+shield eff is dropped.
        State after test_03: 1H(avg=15) + sh(ac=20). eff(1H+sh)=325.
        2H avg=30, eff=224 < 325 -> dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # mindam=25, maxdam=35 -> avg=30, eff=224 < eff(1H+sh)=325 -> dropped
        self._gift_2h_sword(seed=0xDEAD000B, name="Weak 2H", mindam=25, maxdam=35)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0001, "1H unchanged")
        s.assert_body(HR, 0xDEAD0002, "shield unchanged")
        s.assert_absent(0xDEAD000B, "weak 2H dropped")

    def test_05_2h_sword_equips_keeps_insurance(self):
        """Stronger 2H sword equips; old 1H and shield kept as eviction insurance.
        State after test_04: 1H(avg=15) + sh(ac=20), eff=325.
        2H avg=100, eff=742 > 325 -> equips. Old 1H and shield stored as bfi eviction insurance."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # mindam=90, maxdam=110 -> avg=100, eff=742 > eff(1H+sh)=325 -> equips
        self._gift_2h_sword(seed=0xDEAD0003, name="Test 2H Sword", mindam=90, maxdam=110)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0003, "2H equipped")
        assert HR not in s.body, f"HAND_RIGHT should be empty, got {s.body.get(HR):#010x}"
        s.assert_present(0xDEAD0001, "old 1H kept as eviction insurance")
        s.assert_present(0xDEAD0002, "old shield kept as eviction insurance")

    def test_06_weaker_sword_dropped(self):
        """A sword weaker than the active 2H is dropped immediately.
        State after test_05: HAND_LEFT=0xDEAD0003 (2H score 100), HAND_RIGHT=empty,
        bfi=[0xDEAD0001, 0xDEAD0002] as eviction insurance."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # mindam=5, maxdam=10 -> avg=7.5 << 100 -> not better -> dropped
        self._gift_sword(seed=0xDEAD0004, name="Weak Sword", mindam=5, maxdam=10)
        self._settle()
        s = self._state()

        s.assert_present(0xDEAD0001, "old 1H kept as eviction insurance")
        s.assert_present(0xDEAD0002, "old shield kept as eviction insurance")

        s.assert_body(HL, 0xDEAD0003, "2H unchanged")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_absent(0xDEAD0004, "weak sword dropped")

    def test_07_better_1h_upgrades_backup(self):
        """A 1H better than backup[0] but worse than the 2H upgrades backup[0].
        State after test_06: HAND_LEFT=0xDEAD0003 (2H eff=742),
        bfi=[0xDEAD0001(avg=15), 0xDEAD0002(ac=20)].
        New 1H avg=20: combo_eff(20+sh20)=431 < 2H_eff=742, solo_eff(20)=150 > solo_eff(15)=113
        -> upgrades backup[0]. Old 0xDEAD0001 dropped, 0xDEAD000A becomes backup[0]."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # avg=20: combo_eff(431) < 2H_eff(742), solo_eff(150) > bk_solo_eff(113) -> upgrade backup[0]
        self._gift_sword(seed=0xDEAD000A, name="Mid Sword", mindam=15, maxdam=25)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0003, "2H unchanged")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_present(0xDEAD000A, "mid sword becomes backup[0]")
        s.assert_absent(0xDEAD0001, "old backup weapon dropped")
        s.assert_present(0xDEAD0002, "backup shield unchanged")
        bfi = self.agent._backup_for_item.get(0xDEAD0003, [])
        assert bfi == [0xDEAD000A, 0xDEAD0002], \
            f"backup_for_item wrong: {[f'{x:#010x}' for x in bfi]}"

    def test_08_shield_upgrades_backup(self):
        """A shield better than backup[1] but not strong enough to evict upgrades backup[1].
        State after test_07: HAND_LEFT=0xDEAD0003 (2H eff=742),
        bfi=[0xDEAD000A(avg=20), 0xDEAD0002(ac=20)].
        Shield ac=25: combo_eff(20+sh25)=538 < 2H_eff=742, solo_eff(sh25)=2660 > solo_eff(sh20)=2128
        -> upgrades backup[1]. Old 0xDEAD0002 dropped, 0xDEAD000C becomes backup[1]."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # ac=25: combo_eff(538) < 2H_eff(742), solo_eff(sh25)=2660 > solo_eff(sh20)=2128 -> upgrade backup[1]
        self._gift_shield(seed=0xDEAD000C, name="Better Shield", ac=25)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0003, "2H unchanged")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_present(0xDEAD000C, "better shield becomes backup[1]")
        s.assert_absent(0xDEAD0002, "old backup shield dropped")
        s.assert_present(0xDEAD000A, "backup weapon unchanged")
        bfi = self.agent._backup_for_item.get(0xDEAD0003, [])
        assert bfi == [0xDEAD000A, 0xDEAD000C], \
            f"backup_for_item wrong: {[f'{x:#010x}' for x in bfi]}"

    def test_09_shield_blocked_by_2h(self):
        """A shield too weak to beat the 2H via combo is dropped.
        State after test_08: HAND_LEFT=0xDEAD0003 (2H eff=742),
        bfi=[0xDEAD000A(avg=20), 0xDEAD000C(ac=25)].
        Shield ac=10: combo_eff(20+sh10)=294 < 2H_eff=742 and solo_eff(sh10)=1450 < solo_eff(sh25)=2660
        -> shield dropped (no eviction, no backup upgrade)."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        self._gift_shield(seed=0xDEAD0009, name="Blocked Shield", ac=10)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0003, "2H unchanged")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_absent(0xDEAD0009, "weak shield blocked and dropped")
        s.assert_present(0xDEAD000A, "backup 1H still in inv")
        s.assert_present(0xDEAD000C, "backup shield still in inv")

    def test_10_shield_evicts_2h_via_insurance(self):
        """A shield that pushes combo eff above 2H eff evicts the 2H.
        State after test_09: HAND_LEFT=0xDEAD0003 (2H eff=742),
        bfi=[0xDEAD000A(avg=20), 0xDEAD000C(ac=25)].
        Shield ac=100: combo_eff(20+sh100)=2153 > 2H_eff=742 -> evicts 2H.
        0xDEAD000A equips to HAND_LEFT, shield to HAND_RIGHT, 0xDEAD000C and 2H dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        self._gift_shield(seed=0xDEAD0005, name="Strong Shield", ac=100)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD000A, "backup 1H equips to HAND_LEFT")
        s.assert_body(HR, 0xDEAD0005, "stronger shield equipped")
        s.assert_absent(0xDEAD0003, "2H dropped after eviction")
        s.assert_absent(0xDEAD000C, "old backup shield dropped (non-partner backup)")

    def test_11_1h_replaces_1h(self):
        """A stronger 1H sword equips over the existing 1H; old 1H dropped.
        State after test_10: HAND_LEFT=0xDEAD000A (avg=20), HAND_RIGHT=0xDEAD0005 (ac=100).
        new 1H avg=75: eff(75+sh100)=7984 >> eff(20+sh100)=2153 -> equips."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value

        # mindam=65, maxdam=85 -> avg=75; eff(75+sh100)=7984 >> eff(20+sh100)=2153 -> equips
        self._gift_sword(seed=0xDEAD0006, name="Strong 1H", mindam=65, maxdam=85)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0006, "stronger 1H equipped")
        s.assert_body(HR, 0xDEAD0005, "shield still equipped")
        s.assert_absent(0xDEAD000A, "displaced 1H dropped")

    def test_12_minstr_sword_dropped(self):
        """A sword whose STR requirement (100) exceeds the player's STR is dropped.
        State after test_11: HAND_LEFT=0xDEAD0006 (1H), HAND_RIGHT=0xDEAD0005 (shield)."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value

        # minstr=100 > warrior base STR ~30-50 -> _can_equip fails -> dropped
        # (100 fits in int8; 255 would overflow the signed field)
        _gift(self.game, seed=0xDEAD0007,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name="Req Sword", mindam=100, maxdam=150, minstr=100)
        self._settle()
        s = self._state()

        s.assert_body(HL, 0xDEAD0006, "equipped sword unchanged")
        s.assert_absent(0xDEAD0007, "high-minstr sword dropped")

    def test_13_potion_kept(self):
        """A gifted healing potion is kept (belt or inventory) without being dropped.
        State after test_12: HAND_LEFT=0xDEAD0006, HAND_RIGHT=0xDEAD0005."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value

        self._gift_potion(seed=0xDEAD0008, name="Test Potion")
        self._settle()
        s = self._state()

        s.assert_present(0xDEAD0008, "healing potion kept in belt/inv")
        s.assert_body(HL, 0xDEAD0006, "equipped sword unchanged")

    def run_all(self):
        # Tests run sequentially and share engine state: each test leaves the
        # world in a known state that the next test builds on. Stop on first
        # failure to avoid cascading errors from a broken baseline.
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv2HBackupTests:
    """2H backup-restore scenario.
    Tests the full lifecycle: unidentified 2H equips (backup set), weaker 1H
    dropped (combo with backup shield still loses), stronger 1H wins (1H equips,
    backup shield restored, old 1H and 2H both destroyed)."""

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL = None   # HAND_LEFT cii
        self._HR = None   # HAND_RIGHT cii
        self._s0 = None   # baseline after test_01
        self._2h_seed      = 0xBEEF0001
        self._bk_sw_seed   = None   # old 1H seed saved as backup
        self._bk_sh_seed   = None   # old shield seed saved as backup

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_sword(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def _gift_shield(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac)

    def _gift_2h_unidentified(self, seed, name, mindam, maxdam):
        """Unidentified 2H: _iMagical=MAGIC so _needs_identification returns True."""
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def test_01_initial_state(self):
        """Verify clean start: 1H weapon + shield."""
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        self._s0 = self._state()
        assert self._HL in self._s0.body, "HAND_LEFT should have a weapon"
        assert self._HR in self._s0.body, "HAND_RIGHT should have a shield"
        self._bk_sw_seed = self._s0.body[self._HL]
        self._bk_sh_seed = self._s0.body[self._HR]
        print(f"  1H seed={self._bk_sw_seed:#010x}  shield seed={self._bk_sh_seed:#010x}")

    def test_02_unidentified_2h_equips_backup_set(self):
        """Unidentified 2H (avg=15, eff=113) equips; starting 1H and shield reserved as backup.
        eff_2H(15)=113 > eff_starter(ss+bk)=53. Use avg=15 so inheritance tests can evict it."""
        HL = self._HL
        HR = self._HR
        # avg=15, eff=113 > eff_starter=53 -> 2H equips; starter combo is backup.
        self._gift_2h_unidentified(self._2h_seed, "Test 2H", mindam=10, maxdam=20)
        self._settle()
        s = self._state()

        s.assert_body(HL, self._2h_seed, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H active"
        s.assert_present(self._bk_sw_seed, "old 1H kept as backup")
        s.assert_present(self._bk_sh_seed, "old shield kept as backup")
        bfi = self.agent._backup_for_item.get(self._2h_seed, [])
        assert bfi == [self._bk_sw_seed, self._bk_sh_seed], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_03_weaker_shield_dropped(self):
        """Shield whose solo score <= backup[1] solo score is dropped; backup unchanged.
        AC=1 -> score=0.5, well below the starting shield AC."""
        HL = self._HL
        self._gift_shield(seed=0xBEEF0010, name="Weak Shield", ac=1)
        self._settle()
        s = self._state()

        s.assert_body(HL, self._2h_seed, "2H unchanged")
        s.assert_absent(0xBEEF0010, "weak shield dropped")
        s.assert_present(self._bk_sw_seed, "backup 1H still in inv")
        s.assert_present(self._bk_sh_seed, "backup shield still in inv")
        bfi = self.agent._backup_for_item.get(self._2h_seed, [])
        assert bfi == [self._bk_sw_seed, self._bk_sh_seed], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_04_weaker_1h_dropped(self):
        """1H whose solo score <= backup[0] solo score is dropped; backup unchanged.
        mindam=0, maxdam=0 -> score=0.0, below any real starter sword."""
        HL = self._HL
        self._gift_sword(seed=0xBEEF0011, name="Junk 1H", mindam=0, maxdam=0)
        self._settle()
        s = self._state()

        s.assert_body(HL, self._2h_seed, "2H unchanged")
        s.assert_absent(0xBEEF0011, "junk 1H dropped")
        s.assert_present(self._bk_sw_seed, "backup 1H still in inv")
        s.assert_present(self._bk_sh_seed, "backup shield still in inv")
        bfi = self.agent._backup_for_item.get(self._2h_seed, [])
        assert bfi == [self._bk_sw_seed, self._bk_sh_seed], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_05_weaker_1h_upgrades_backup(self):
        """1H with solo score > backup[0] upgrades the backup weapon slot.
        The old starter sword is dropped; weak 1H takes its place in backup[0].
        The 2H stays equipped (weak 1H solo score 5.5 still loses to 2H score 60)."""
        HL = self._HL
        self._gift_sword(seed=0xBEEF0002, name="Weak 1H", mindam=1, maxdam=10)
        self._settle()
        s = self._state()

        s.assert_body(HL, self._2h_seed, "2H unchanged")
        s.assert_present(0xBEEF0002, "weak 1H now in backup[0]")
        s.assert_absent(self._bk_sw_seed, "old starter sword replaced in backup")
        s.assert_present(self._bk_sh_seed, "backup shield still in inv")
        bfi = self.agent._backup_for_item.get(self._2h_seed, [])
        assert bfi == [0xBEEF0002, self._bk_sh_seed], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_06_stronger_1h_restores_backup_shield(self):
        """1H whose weapon-only score > 2H score equips regardless of shield AC.
        2H avg=60; use mindam=65, maxdam=75 -> avg=70 > 60 even without shield.
        Expected: 1H equips, backup shield restored to HAND_RIGHT,
        backup[0] (0xBEEF0002 from test_03) destroyed, 2H destroyed."""
        HL = self._HL
        HR = self._HR
        self._gift_sword(seed=0xBEEF0003, name="Strong 1H", mindam=65, maxdam=75)
        self._settle(max_ticks=30)
        s = self._state()

        s.assert_body(HL, 0xBEEF0003,       "new 1H equipped")
        s.assert_body(HR, self._bk_sh_seed,  "backup shield restored")
        s.assert_absent(0xBEEF0002,          "old backup[0] weapon destroyed")
        s.assert_absent(self._2h_seed,       "displaced 2H destroyed")

    def run_all(self):
        # Sequential: each test builds on the state left by the previous one.
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class InvIdentifyTests:
    """Identification path for an equipped unidentified non-2H weapon.
    cursed=False (path a): identified 1H beats backup -> backup dropped, 1H stays.
    cursed=True  (path b): identified 1H is cursed (score<0) -> dropped, backup re-equipped."""

    _1H_SEED     = 0xFEED0001
    _1H_SEED2    = 0xFEED0003
    _SCROLL_SEED = 0xFEED0002

    def __init__(self, game, cursed=False):
        self.game    = game
        self._cursed = cursed
        stand        = _StandModel()
        runners      = {lvl: stand for lvl in range(1, 17)}
        self.agent   = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL = None
        self._HR = None
        self._ss_seed = None   # Short Sword seed (stashed as backup)
        self._bk_seed = None   # Buckler seed (stays in HAND_RIGHT throughout)

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_1h(self, seed, name, mindam, maxdam, pldam_mod=0):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              pldam_mod=pldam_mod)

    def _gift_scroll_of_identify(self, seed):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_SCROLL_OF,
              name="Scroll of Identify",
              mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_SCROLL.value,
              ididx=_IDI_IDENTIFY,
              ispell=dx.SpellID.Identify.value)

    def test_01_initial_state(self):
        """Verify clean start: 1H weapon + shield."""
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._ss_seed = s.body[self._HL]
        self._bk_seed = s.body[self._HR]
        print(f"  1H seed={self._ss_seed:#010x}  shield seed={self._bk_seed:#010x}")

    def test_02_1h_equips_backup_set(self):
        """Unidentified 1H (avg=60 >> Short Sword avg=4) equips; Short Sword
        stashed as backup for the unidentified occupant."""
        HL = self._HL
        HR = self._HR
        pldam_mod = -1000 if self._cursed else 0
        self._gift_1h(self._1H_SEED, "Test 1H", mindam=50, maxdam=70, pldam_mod=pldam_mod)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._1H_SEED, "1H equipped")
        s.assert_body(HR, self._bk_seed, "Buckler still in HAND_RIGHT")
        s.assert_present(self._ss_seed, "Short Sword kept as backup")
        bfi = self.agent._backup_for_item.get(self._1H_SEED, [])
        assert bfi == [self._ss_seed], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_03_stronger_1h_takes_over(self):
        """A second stronger unidentified 1H arrives while the first is equipped.
        The backup (Short Sword) is dropped; the weaker first 1H is a new backup."""
        HL = self._HL
        HR = self._HR
        pldam_mod = -1000 if self._cursed else 0
        self._gift_1h(self._1H_SEED2, "Stronger 1H", mindam=70, maxdam=90, pldam_mod=pldam_mod)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._1H_SEED2, "stronger 1H now equipped")
        s.assert_body(HR, self._bk_seed,  "Buckler unchanged")
        s.assert_present(self._1H_SEED,   "weaker first 1H displaced but kept as a backup")
        s.assert_absent(self._ss_seed,    "Short Sword backup lost compared to previous 1H and dropped")
        bfi = self.agent._backup_for_item.get(self._1H_SEED2, [self._1H_SEED])
        assert bfi == [self._1H_SEED], \
            f"backup_for_item wrong: {[f'{s:#010x}' for s in bfi]}"

    def test_04_identify_1h(self):
        """Gift a scroll of identify; agent identifies the currently equipped 1H (the stronger one).
        Path a (cursed=False): score=80 > Short Sword backup -> 1H wins, backup dropped.
        Path b (cursed=True):  identified score<0 -> cursed, dropped, Short Sword restored."""
        HL = self._HL
        HR = self._HR
        self._gift_scroll_of_identify(self._SCROLL_SEED)
        self._settle(max_ticks=30)
        s = self._state()
        if not self._cursed:
            # Path a: 1H wins over weaker first 1H; backup dropped.
            s.assert_body(HL, self._1H_SEED2, "1H still equipped after identification")
            s.assert_body(HR, self._bk_seed,  "Buckler unchanged")
            s.assert_absent(self._ss_seed,    "Short Sword backup dropped")
            s.assert_absent(self._1H_SEED,    "weaker first 1H dropped")
        else:
            # Path b: 1H is cursed (score<0), dropped; Short Sword restored.
            s.assert_body(HL, self._1H_SEED,  "weaker prev 1H restored to HAND_LEFT")
            s.assert_body(HR, self._bk_seed,  "Buckler unchanged")
            s.assert_absent(self._1H_SEED2,   "cursed 1H dropped")

    def run_all(self):
        label = "cursed" if self._cursed else "normal"
        print(f"\n--- InvIdentifyTests ({label}) ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv2HIdentifyTests:
    """Identification path for an equipped unidentified 2H.
    cursed=False (path a): identified 2H beats backup combo -> backup kept as eviction insurance.
    cursed=True  (path b): identified 2H is cursed (score<0) -> dropped, backup restored."""

    _2H_SEED     = 0xCAFE0001
    _2H_SEED2    = 0xCAFE0003
    _SCROLL_SEED = 0xCAFE0002

    def __init__(self, game, cursed=False):
        self.game    = game
        self._cursed = cursed
        stand        = _StandModel()
        runners      = {lvl: stand for lvl in range(1, 17)}
        self.agent   = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL = None
        self._HR = None
        self._bk_sw_seed = None
        self._bk_sh_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_2h(self, seed, name, mindam, maxdam, pldam_mod=0):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              pldam_mod=pldam_mod)

    def _gift_scroll_of_identify(self, seed):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_SCROLL_OF,
              name="Scroll of Identify",
              mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_SCROLL.value,
              ididx=_IDI_IDENTIFY,
              ispell=dx.SpellID.Identify.value)

    def test_01_initial_state(self):
        """Verify clean start: 1H weapon + shield."""
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._bk_sw_seed = s.body[self._HL]
        self._bk_sh_seed = s.body[self._HR]
        print(f"  1H seed={self._bk_sw_seed:#010x}  shield seed={self._bk_sh_seed:#010x}")

    def test_02_2h_equips_backup_set(self):
        """Unidentified 2H equips; old 1H and shield reserved as backup."""
        HL = self._HL
        HR = self._HR
        # pldam_mod=-1000 makes identified score = (60 + (-1000)) * 1.0 = -940 for cursed variant.
        pldam_mod = -1000 if self._cursed else 0
        self._gift_2h(self._2H_SEED, "Test 2H", mindam=50, maxdam=70, pldam_mod=pldam_mod)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H active"
        s.assert_present(self._bk_sw_seed, "old 1H kept as backup")
        s.assert_present(self._bk_sh_seed, "old shield kept as backup")

    def test_03_stronger_2h_takes_over(self):
        """A second stronger unidentified 2H arrives while the first is equipped.
        Stronger 2H inherits all the backups."""
        HL = self._HL
        pldam_mod = -1000 if self._cursed else 0
        self._gift_2h(self._2H_SEED2, "Stronger 2H", mindam=70, maxdam=90, pldam_mod=pldam_mod)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._2H_SEED2,   "stronger 2H now equipped")
        s.assert_present(self._bk_sw_seed,  "old 1H backup inherited")
        s.assert_present(self._bk_sh_seed,  "old shield backup inherited")
        s.assert_absent(self._2H_SEED,      "weaker first 2H displaced and dropped")

    def test_04_identify_2h(self):
        """Gift a scroll of identify; agent identifies the currently equipped 2H (the stronger one).
        Path a (cursed=False): score=80; no live backups remain -> 2H stays, nothing restored.
        Path b (cursed=True):  identified score<0 -> cursed, dropped; backups restored."""
        HL = self._HL
        HR = self._HR
        self._gift_scroll_of_identify(self._SCROLL_SEED)
        self._settle(max_ticks=30)
        s = self._state()
        if not self._cursed:
            # Stronger 2H stays, all backups kept.
            s.assert_body(HL, self._2H_SEED2,  "stronger 2H stays after identification")
            s.assert_present(self._bk_sw_seed,  "old 1H backup absent")
            s.assert_present(self._bk_sh_seed,  "old shield backup absent")
        else:
            # Cursed 2H dropped; backups restored.
            s.assert_body(HL, self._bk_sw_seed, "old 1H restored")
            s.assert_body(HR, self._bk_sh_seed, "old shield restored")
            s.assert_absent(self._2H_SEED2,    "cursed 2H dropped")

    def run_all(self):
        label = "cursed" if self._cursed else "normal"
        print(f"\n--- Inv2HIdentifyTests ({label}) ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class _Inv2HWithBackupBase:
    """Base for scenarios starting with: unidentified 2H equipped,
    starting 1H (ss) and shield (bk) reserved as backup[0] and backup[1]."""

    _2H_SEED = None

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL      = None
        self._HR      = None
        self._ss_seed = None
        self._bk_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_2h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_1h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_shield_unidentified(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def test_01_initial_state(self):
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._ss_seed = s.body[self._HL]
        self._bk_seed = s.body[self._HR]
        print(f"  ss={self._ss_seed:#010x}  bk={self._bk_seed:#010x}")

    def test_02_2h_equips_backup_set(self):
        """Unidentified 2H (avg=15, eff=113) equips; starting 1H and shield reserved as backup.
        eff_2H(15)=113 > eff_starter(ss+bk)=53. Use avg=15 so inheritance tests can evict it."""
        HL = self._HL
        HR = self._HR
        # avg=15, eff=113 > eff_starter=53 -> 2H equips; starter combo is backup.
        self._gift_2h_unidentified(self._2H_SEED, "First 2H", mindam=10, maxdam=20)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H active"
        s.assert_present(self._ss_seed, "ss kept as backup[0]")
        s.assert_present(self._bk_seed, "bk kept as backup[1]")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._ss_seed, self._bk_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"

    def run_all(self):
        label = getattr(self, '_label', type(self).__name__)
        print(f"\n--- {label} ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv2HTo2HInheritanceTests(_Inv2HWithBackupBase):
    """Scenario 1a: stronger unidentified 2H evicts 2H and inherits [ss, bk]."""
    _label    = "Inv2HTo2HInheritanceTests"
    _2H_SEED  = 0xA1000001
    _2H_SEED2 = 0xA1000002

    def test_03_stronger_2h_inherits_backup(self):
        # eff(new 2H avg=80)=593 > eff(old 2H avg=15)=113
        HL = self._HL
        self._gift_2h_unidentified(self._2H_SEED2, "Stronger 2H", mindam=70, maxdam=90)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._2H_SEED2, "stronger 2H equipped")
        s.assert_absent(self._2H_SEED,    "old 2H dropped")
        s.assert_present(self._ss_seed,   "ss inherited as backup[0]")
        s.assert_present(self._bk_seed,   "bk inherited as backup[1]")
        bfi = self.agent._backup_for_item.get(self._2H_SEED2, [])
        assert bfi == [self._ss_seed, self._bk_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class Inv2HTo1HInheritanceTests(_Inv2HWithBackupBase):
    """Scenario 1b: stronger unidentified 1H evicts 2H; ss inherited as backup."""
    _label   = "Inv2HTo1HInheritanceTests"
    _2H_SEED = 0xA2000001
    _1H_SEED = 0xA2000002

    def test_03_stronger_1h_inherits_backup_weapon(self):
        # combo_eff(1H avg=60 + bk3)=737 > eff(2H avg=15)=113
        HL = self._HL
        HR = self._HR
        self._gift_1h_unidentified(self._1H_SEED, "Stronger 1H", mindam=55, maxdam=65)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._1H_SEED, "1H equipped")
        s.assert_body(HR, self._bk_seed, "bk equips as partner")
        s.assert_absent(self._2H_SEED,   "2H dropped")
        s.assert_present(self._ss_seed,  "ss inherited as backup for 1H")
        bfi = self.agent._backup_for_item.get(self._1H_SEED, [])
        assert bfi == [self._ss_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class Inv2HToShInheritanceTests(_Inv2HWithBackupBase):
    """Scenario 1c: stronger unidentified shield evicts 2H; bk inherited as backup."""
    _label   = "Inv2HToShInheritanceTests"
    _2H_SEED = 0xA3000001
    _SH_SEED = 0xA3000002

    def test_03_stronger_shield_inherits_backup_shield(self):
        # combo_eff(ss avg=4 + sh120)=456 > eff(2H avg=15)=113
        HL = self._HL
        HR = self._HR
        self._gift_shield_unidentified(self._SH_SEED, "Strong Shield", ac=120)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._ss_seed, "ss equips as partner")
        s.assert_body(HR, self._SH_SEED, "shield equipped")
        s.assert_absent(self._2H_SEED,   "2H dropped")
        s.assert_present(self._bk_seed,  "bk inherited as backup for shield")
        bfi = self.agent._backup_for_item.get(self._SH_SEED, [])
        assert bfi == [self._bk_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class _Inv1HWithBackupBase:
    """Base for scenarios starting with: unidentified 1H equipped,
    starting sword (ss) reserved as backup[0], buckler (bk) on HAND_RIGHT."""

    _1H_SEED = None

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL      = None
        self._HR      = None
        self._ss_seed = None
        self._bk_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_1h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_2h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_2h_identified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def test_01_initial_state(self):
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._ss_seed = s.body[self._HL]
        self._bk_seed = s.body[self._HR]
        print(f"  ss={self._ss_seed:#010x}  bk={self._bk_seed:#010x}")

    def test_02_1h_equips_backup_set(self):
        """Unidentified 1H (avg=60) equips; starting sword stashed as backup."""
        HL = self._HL
        HR = self._HR
        self._gift_1h_unidentified(self._1H_SEED, "First 1H", mindam=50, maxdam=70)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._1H_SEED, "1H equipped")
        s.assert_body(HR, self._bk_seed, "buckler unchanged")
        s.assert_present(self._ss_seed,  "ss stashed as backup")
        bfi = self.agent._backup_for_item.get(self._1H_SEED, [])
        assert bfi == [self._ss_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"

    def run_all(self):
        label = getattr(self, '_label', type(self).__name__)
        print(f"\n--- {label} ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv1HTo2HInheritanceTests(_Inv1HWithBackupBase):
    """Scenario 2a: 2H evicts 1H+shield; old 1H and bk become backup (unidentified 2H only)."""

    _1H_SEED = 0xA4000001
    _2H_SEED = 0xA4000002

    def __init__(self, game, identified=False):
        super().__init__(game)
        self._identified = identified
        self._label = ("Inv1HTo2HInheritanceTests "
                       f"({'identified' if identified else 'unidentified'})")

    def test_03_2h_takes_over_1h(self):
        # eff_2H(avg=110)=816 > eff(1H_avg60+bk_ac3)=737 -> 2H evicts
        HL = self._HL
        HR = self._HR
        if self._identified:
            self._gift_2h_identified(self._2H_SEED, "Strong 2H", mindam=100, maxdam=120)
        else:
            self._gift_2h_unidentified(self._2H_SEED, "Strong 2H", mindam=100, maxdam=120)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H active"
        s.assert_absent(self._ss_seed,  "old backup ss dropped")
        s.assert_present(self._1H_SEED, "1H preserved as backup[0]")
        s.assert_present(self._bk_seed, "bk preserved as backup[1]")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._1H_SEED, self._bk_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class Inv1HTo1HInheritanceTests(_Inv1HWithBackupBase):
    """Scenario 2b: stronger unidentified 1H evicts 1H; old 1H becomes backup, ss dropped."""
    _label    = "Inv1HTo1HInheritanceTests"
    _1H_SEED  = 0xA5000001
    _1H_SEED2 = 0xA5000002

    def test_03_stronger_1h_makes_old_1h_backup(self):
        """Stronger unidentified 1H (avg=80) evicts current unidentified 1H.
        Old 1H becomes backup for new 1H. ss dropped. bk unchanged."""
        HL = self._HL
        HR = self._HR
        self._gift_1h_unidentified(self._1H_SEED2, "Stronger 1H", mindam=70, maxdam=90)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._1H_SEED2, "stronger 1H equipped")
        s.assert_body(HR, self._bk_seed,   "buckler unchanged")
        s.assert_absent(self._ss_seed,     "old backup ss dropped")
        s.assert_present(self._1H_SEED,    "old 1H preserved as backup")
        bfi = self.agent._backup_for_item.get(self._1H_SEED2, [])
        assert bfi == [self._1H_SEED], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class _Inv1HShWithBackupBase:
    """Base for scenarios starting with: 1H (ss) on HAND_LEFT and unidentified
    shield (sh_main) on HAND_RIGHT, old shield (bk) reserved as backup for sh_main."""

    _SH_SEED = None

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._HL      = None
        self._HR      = None
        self._ss_seed = None   # starting 1H on HAND_LEFT
        self._bk_seed = None   # old shield, becomes backup for sh_main

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_shield_unidentified(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_2h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_2h_identified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def test_01_initial_state(self):
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._ss_seed = s.body[self._HL]
        self._bk_seed = s.body[self._HR]
        print(f"  ss={self._ss_seed:#010x}  bk={self._bk_seed:#010x}")

    def test_02_sh_equips_backup_set(self):
        """Unidentified shield (ac=10) equips; old shield stashed as backup."""
        HL = self._HL
        HR = self._HR
        self._gift_shield_unidentified(self._SH_SEED, "Main Shield", ac=10)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._ss_seed, "ss unchanged on HAND_LEFT")
        s.assert_body(HR, self._SH_SEED, "unidentified shield equipped")
        s.assert_present(self._bk_seed,  "old bk stashed as backup")
        bfi = self.agent._backup_for_item.get(self._SH_SEED, [])
        assert bfi == [self._bk_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"

    def run_all(self):
        label = getattr(self, '_label', type(self).__name__)
        print(f"\n--- {label} ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv1HShTo2HInheritanceTests(_Inv1HShWithBackupBase):
    """Scenario 3a: 2H evicts 1H+shield; ss and sh_main become backup, old bk dropped."""

    _SH_SEED = 0xA6000001
    _2H_SEED = 0xA6000002

    def __init__(self, game, identified=False):
        super().__init__(game)
        self._identified = identified
        self._label = ("Inv1HShTo2HInheritanceTests "
                       f"({'identified' if identified else 'unidentified'})")

    def test_03_2h_takes_over_sh(self):
        # weapon_only(2H avg=35) > combo(ss_avg~4.5 + sh_main_ac10*0.5=5) ~= 9.5
        HL = self._HL
        HR = self._HR
        if self._identified:
            self._gift_2h_identified(self._2H_SEED, "Strong 2H", mindam=30, maxdam=40)
        else:
            self._gift_2h_unidentified(self._2H_SEED, "Strong 2H", mindam=30, maxdam=40)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H active"
        s.assert_absent(self._bk_seed,   "old backup bk dropped")
        s.assert_present(self._ss_seed,  "ss preserved as backup[0]")
        s.assert_present(self._SH_SEED,  "sh_main preserved as backup[1]")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._ss_seed, self._SH_SEED], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class Inv1HShToShInheritanceTests(_Inv1HShWithBackupBase):
    """Scenario 3b: stronger unidentified shield evicts sh_main; sh_main becomes backup, bk dropped."""
    _label    = "Inv1HShToShInheritanceTests"
    _SH_SEED  = 0xA7000001
    _SH_SEED2 = 0xA7000002

    def test_03_stronger_sh_makes_old_sh_backup(self):
        # shield_score(ac=20*0.5=10) > shield_score(ac=10*0.5=5)
        HL = self._HL
        HR = self._HR
        self._gift_shield_unidentified(self._SH_SEED2, "Stronger Shield", ac=20)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(HL, self._ss_seed,  "ss unchanged on HAND_LEFT")
        s.assert_body(HR, self._SH_SEED2, "stronger shield equipped")
        s.assert_absent(self._bk_seed,    "old backup bk dropped")
        s.assert_present(self._SH_SEED,   "old sh_main preserved as backup")
        bfi = self.agent._backup_for_item.get(self._SH_SEED2, [])
        assert bfi == [self._SH_SEED], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"


class InvDurabilityTests:
    """Durability tiebreak: among equal-score items, the higher-durability one is kept.
    Note: maxdur=255 is DUR_INDESTRUCTIBLE; _dur_ratio() returns 1.0 for it regardless
    of _iDurability. Use maxdur<255 for meaningful ratio comparisons.
    ChangeEquipment() copies the item as-is (no normalization), so a worn sword gifted
    with durability<maxdur stays worn after equipping - preserved in InvBody."""
    _SWORD_WORN  = 0xDEB00001  # worn (dur=50/100) - equips first
    _SWORD_FULL  = 0xDEB00002  # same score, full (dur=100/100) - wins tiebreak
    _SWORD_LOWER = 0xDEB00003  # same score, more worn (dur=30/100) - loses tiebreak

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True

    def _settle(self):
        _settle(self.agent)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def test_01_initial_state(self):
        """Verify clean start: 1H+shield equipped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert HL in s.body, "HAND_LEFT should have a weapon"
        assert HR in s.body, "HAND_RIGHT should have a shield"

    def test_02_worn_sword_equips(self):
        """Worn sword (dur=50/100, score=15) beats the starter -> equips.
        Engine copies item as-is so InvBody retains durability=50."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        _gift(self.game, seed=self._SWORD_WORN,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name="Sword Worn", mindam=10, maxdam=20,
              durability=50, maxdur=100)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._SWORD_WORN, "worn sword equipped")

    def test_03_full_sword_wins_tiebreak(self):
        """Full-durability sword (dur=100/100) wins tiebreak over worn (dur=50/100) -> equips."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        _gift(self.game, seed=self._SWORD_FULL,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name="Sword Full", mindam=10, maxdam=20,
              durability=100, maxdur=100)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._SWORD_FULL, "full sword won tiebreak")
        s.assert_absent(self._SWORD_WORN, "worn sword dropped")

    def test_04_lower_sword_loses_tiebreak(self):
        """More-worn sword (dur=30/100) loses tiebreak vs full (dur=100/100) -> dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        _gift(self.game, seed=self._SWORD_LOWER,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name="Sword Lower", mindam=10, maxdam=20,
              durability=30, maxdur=100)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._SWORD_FULL, "full sword still equipped")
        s.assert_absent(self._SWORD_LOWER, "lower-durability sword dropped")

    def run_all(self):
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv2HStaleBackupTests:
    """Stale backup weapon scenarios.
    When the backup weapon is externally destroyed, _cleanup_stale_backups trims it
    from bfi, leaving a lone shield at bfi[0]. A subsequent stronger shield upgrades
    that lone shield backup (best-available policy). The 2H is NOT evicted because
    eviction via the phantom-weapon (zero-score) path is blocked by the class check."""

    _2H_SEED    = 0xF00D0001
    _SH2_SEED   = 0xF00D0002   # strong shield used in test_04
    _WP2_SEED   = 0xF00D0003   # weapon used in test_05
    _WP3_SEED   = 0xF00D0004   # weapon used in test_06

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._bk_sw_seed = None
        self._bk_sh_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_2h_unidentified(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value)

    def _gift_shield(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac)

    def _gift_1h(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def _find_inv_cii(self, seed):
        """Return the CII of an InvList item with the given seed, or None."""
        player    = self.game.safe_state.player
        inv_first = dx.inv_item.INVITEM_INV_FIRST.value
        for i in range(int(player._pNumInv)):
            if int(player.InvList[i]._iSeed) == seed:
                return inv_first + i
        return None

    def _destroy_item(self, cii):
        """Submit a drop+destroy ring command for the given CII."""
        RE = ring.RingEntryType
        self.game.submit_key(
            RE.RING_ENTRY_KEY_INV_DROP_ITEM | RE.RING_ENTRY_F_SINGLE_TICK_PRESS,
            data=(cii, 2))   # data2 bit1 = destroy

    def test_01_initial_state(self):
        """Verify clean start: 1H+shield in body."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert HL in s.body, "HAND_LEFT should have a weapon"
        assert HR in s.body, "HAND_RIGHT should have a shield"
        self._bk_sw_seed = s.body[HL]
        self._bk_sh_seed = s.body[HR]
        print(f"  1H seed={self._bk_sw_seed:#010x}  shield seed={self._bk_sh_seed:#010x}")

    def test_02_2h_equips_backup_set(self):
        """Unidentified 2H (avg=60) equips; old 1H and shield stored as bfi backups."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._gift_2h_unidentified(self._2H_SEED, "Stale 2H", mindam=50, maxdam=70)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H"
        s.assert_present(self._bk_sw_seed, "backup weapon in inv")
        s.assert_present(self._bk_sh_seed, "backup shield in inv")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._bk_sw_seed, self._bk_sh_seed], \
            f"bfi wrong: {[f'{x:#010x}' for x in bfi]}"

    def test_03_drop_backup_weapon_externally(self):
        """Externally destroy backup weapon; _cleanup_stale_backups trims bfi to [shield]."""
        cii = self._find_inv_cii(self._bk_sw_seed)
        assert cii is not None, "backup weapon not found in InvList"
        self._destroy_item(cii)
        self._settle()
        s = self._state()
        s.assert_absent(self._bk_sw_seed, "backup weapon gone after destroy")
        s.assert_present(self._bk_sh_seed, "backup shield still present")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._bk_sh_seed], \
            f"bfi should be [shield_seed] after weapon removed: {[f'{x:#010x}' for x in bfi]}"

    def test_04_stronger_shield_upgrades_lone_backup(self):
        """Shield arrives when bfi[0] is a lone shield (weapon was destroyed).
        Strong Shield (ac=200, score=100) beats the lone Buckler backup -> upgrades bfi[0].
        2H is NOT evicted (no weapon partner for a combo), Buckler is dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._gift_shield(self._SH2_SEED, "Strong Shield", ac=200)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H still equipped")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_present(self._SH2_SEED, "strong shield replaced backup shield")
        s.assert_absent(self._bk_sh_seed, "old buckler backup dropped")

    def test_05_weapon_restores_backup_pair(self):
        """After test_04: bfi=[Strong Shield] (lone shield). A new weapon arrives.
        Weapon inserts at bfi[0] without dropping the shield - pair is restored."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        # Weapon score must be below the combo (weapon+shield) to not evict the 2H.
        # combo ~ _weapon_only_score(this) + _shield_only_score(Strong Shield=ac200=score100)
        # 2H score = 60, so weapon_only must be < 60-100 = -40 which is impossible to
        # guarantee that way; use a weak weapon so its solo score (< 60) doesn't evict.
        self._gift_1h(self._WP2_SEED, "Backup Weapon", mindam=1, maxdam=5)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H still equipped")
        assert HR not in s.body, "HAND_RIGHT should stay empty"
        s.assert_present(self._WP2_SEED, "new weapon kept as bfi[0]")
        s.assert_present(self._SH2_SEED, "strong shield kept as bfi[1]")
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert bfi == [self._WP2_SEED, self._SH2_SEED], \
            f"bfi should be [weapon, shield]: {[f'{x:#010x}' for x in bfi]}"

    def test_06_strong_1h_evicts_2h_via_restored_pair(self):
        """After test_05: bfi=[WP2, SH2]. A strong 1H arrives.
        weapon_only=30 < 2H score=60, but combo 30+100(SH2)=130 > 60 -> evicts 2H.
        New 1H and SH2 equip; 2H and WP2 (non-partner backup) are both dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._gift_1h(self._WP3_SEED, "Strong 1H", mindam=25, maxdam=35)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._WP3_SEED, "new 1H equipped")
        s.assert_body(HR, self._SH2_SEED, "strong shield equipped")
        s.assert_absent(self._2H_SEED, "2H dropped after eviction")
        s.assert_absent(self._WP2_SEED, "old backup weapon dropped")

    def run_all(self):
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class InvBeatBackupSpamTests:
    """Finding: 'beats backup' printed every tick for a NORMAL 2H that legitimately
    beats its backup combo (no oscillation, but massive log spam and wasted CPU).
    After the first _resolve_identified_equipped call marks the seed resolved in
    _backup_resolved, subsequent ticks must skip re-evaluation."""

    _2H_SEED  = 0xBB000001
    _SH2_SEED = 0xBB000002   # stronger shield that upgrades backup later

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._bk_sw_seed = None
        self._bk_sh_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_1h(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def _gift_shield(self, seed, name, ac):
        _gift(self.game, seed,
              itype=dx.ItemType.Shield.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_ARMOR.value,
              icurs=_ICURS_BUCKLER,
              name=name, mindam=0, maxdam=0, ac=ac)

    def _gift_2h(self, seed, name, mindam, maxdam):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam)

    def test_01_initial_state(self):
        """Record baseline 1H + shield."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert HL in s.body, "HAND_LEFT should have a weapon"
        assert HR in s.body, "HAND_RIGHT should have a shield"
        self._bk_sw_seed = s.body[HL]
        self._bk_sh_seed = s.body[HR]
        print(f"  1H seed={self._bk_sw_seed:#010x}  shield seed={self._bk_sh_seed:#010x}")

    def test_02_normal_2h_equips_marks_resolved(self):
        """NORMAL 2H (eff=76) beats backup combo (eff=53) -> equips.
        After settling, _backup_resolved must contain the 2H seed so that
        _resolve_identified_equipped does NOT fire again on subsequent ticks."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        # avg=10, eff=76 > eff_starter=53 so 2H wins.
        self._gift_2h(self._2H_SEED, "Normal 2H", mindam=5, maxdam=15)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped")
        assert HR not in s.body, "HAND_RIGHT empty with 2H"
        # Run 20 extra ticks - beats-backup must NOT re-fire.
        for _ in range(20):
            self.agent._tick()
        assert self._2H_SEED in self.agent._backup_resolved, \
            "2H seed should be in _backup_resolved after first resolution"

    def test_03_backup_resolved_cleared_on_backup_change(self):
        """A stronger shield upgrades the backup; _backup_resolved must be cleared
        so the new combo can be re-evaluated against the 2H."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        # Strong enough shield to push combo above 2H score (8.0).
        # shield_only = ac * 0.5; need combo > 8.0.
        # Starting backup weapon score ~= (bk_sw stats). Use very high ac to be sure.
        self._gift_shield(self._SH2_SEED, "Super Shield", ac=30)
        self._settle(max_ticks=30)
        # After combo exceeds 2H score, the backup should win and evict the 2H.
        s = self._state()
        s.assert_absent(self._2H_SEED, "2H evicted after combo beats it")
        s.assert_body(HR, self._SH2_SEED, "super shield equipped")

    def run_all(self):
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class Inv2HOscillationTests:
    """Root cause of the 2H oscillation (log seed 841043365):
    Item::clear() only zeroes _itype, leaving _iSeed from the previously-cleared
    item in memory. When a 2H equips and clears HAND_RIGHT, InvBody[HAND_RIGHT]._iSeed
    retains the displaced shield's seed. If a restore later sets
    _pending_equip[HAND_RIGHT] = (shield_seed, ...), the confirmation loop fires
    on the stale _iSeed match (even though _itype == None_), falsely clears the
    pending entry, and the displaced 2H then re-evaluates without a live shield
    complement, sees old_score = weapon_only(1H) instead of the full combo, and
    re-equips. Fix: guard confirmation with _itype != None_ check.

    Timeline with the magic 2H (unidentified=18.0 > combo=5.5, identified=5.0):
      tick 1: scroll acquired, identify action queued and fired
      tick 2: _resolve_identified_equipped fires, restore queued, SW equips
      tick 3: SW confirmed; shield pending NOT falsely confirmed (fix);
              2H evaluates against full combo (5.5), loses (5.0 < 5.5), dropped;
              BK equips
      tick 4: BK confirmed; drop 2H fires -> settled (4 ticks total)

    Without the fix: false shield confirmation at tick 3 leads to 2H re-equipping
    (5.0 > 4.0 weapon-alone), oscillation exceeds max_ticks=8."""

    _2H_SEED     = 0xCC000001
    _SCROLL_SEED = 0xCC000002

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._bk_sw_seed = None
        self._bk_sh_seed = None

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_2h(self, seed, name, mindam, maxdam, pldam_mod=0):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_TWOHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_TWO_HANDED_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              pldam_mod=pldam_mod)

    def _gift_scroll_of_identify(self, seed):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_SCROLL_OF,
              name="Scroll of Identify",
              mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_SCROLL.value,
              ididx=_IDI_IDENTIFY,
              ispell=dx.SpellID.Identify.value)

    def test_01_initial_state(self):
        """Record baseline 1H + shield seeds."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert HL in s.body, "HAND_LEFT should have a weapon"
        assert HR in s.body, "HAND_RIGHT should have a shield"
        self._bk_sw_seed = s.body[HL]
        self._bk_sh_seed = s.body[HR]
        print(f"  1H seed={self._bk_sw_seed:#010x}  shield seed={self._bk_sh_seed:#010x}")

    def test_02_magic_2h_equips_clears_seed_in_hand_right(self):
        """Unidentified magic 2H (unidentified score=18.0 >> combo ~5.5) equips.
        Engine calls InvBody[HAND_RIGHT].clear() which zeroes both _itype and _iSeed.
        Verifies the engine fix: no stale seed remains in the empty slot."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        # unidentified = (14+22)/2 = 18.0 >> combo ~5.5; identified = 18-13 = 5.0 < combo
        self._gift_2h(self._2H_SEED, "Oscillation 2H", mindam=14, maxdam=22, pldam_mod=-13)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._2H_SEED, "2H equipped in left hand")
        assert HR not in s.body, "HAND_RIGHT empty with 2H"
        bfi = self.agent._backup_for_item.get(self._2H_SEED, [])
        assert len(bfi) == 2, f"bfi should hold [weapon, shield]: {[f'{x:#010x}' for x in bfi]}"
        player = self.game.safe_state.player
        slot = player.InvBody[HR]
        assert int(slot._itype) == dx.ItemType.None_.value, "slot must be empty"
        assert int(slot._iSeed) == 0, \
            f"engine fix: _iSeed should be 0 after clear(), got {int(slot._iSeed):#010x}"
        print(f"  confirmed _iSeed=0x00000000 in empty HAND_RIGHT (engine fix)")

    def test_03_identify_2h_restores_1h_shield_without_oscillation(self):
        """Identify the 2H via scroll. Identified score=5.0 < combo ~5.5, so
        restore fires: equip 1H to HAND_LEFT, equip shield to HAND_RIGHT.
        After 1H equips, the confirmation loop runs with HAND_RIGHT still empty
        but InvBody[HAND_RIGHT]._iSeed == shield_seed (stale from 2H equip).
        WITHOUT fix: stale seed falsely confirms _pending_equip[HAND_RIGHT];
          the displaced 2H re-evaluates against 1H alone (no combo), sees
          old_score=4.0 < new_score=5.0, and re-equips - oscillation (exceeds max_ticks).
        WITH fix: _itype guard blocks false confirmation; 2H evaluates against
          full combo (5.5), loses (5.0 < 5.5), and is dropped."""
        HL = dx.inv_item.INVITEM_HAND_LEFT.value
        HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._gift_scroll_of_identify(self._SCROLL_SEED)
        # WITH fix: settles in 6 ticks (identify->restore->equip SW->equip BK->drop 2H).
        # WITHOUT fix: stale-seed false confirmation causes 2H re-equip; BK eviction
        # cycle adds ~5 extra ticks (total ~11), which exceeds max_ticks=8.
        self._settle(max_ticks=8)
        s = self._state()
        s.assert_body(HL, self._bk_sw_seed, "1H weapon restored after 2H identified")
        s.assert_body(HR, self._bk_sh_seed, "shield restored after 2H identified")
        s.assert_absent(self._2H_SEED, "2H dropped after restore")

    def run_all(self):
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class InvUnidentifiedUpgradeTests:
    """Bug: unidentified incoming weapon is compared against the equipped weapon's
    IDENTIFIED score, not its basic score.  When the equipped item is an identified
    magic sword, its old_score includes magic bonuses.  An incoming unidentified sword
    with higher base damage but lower base-than-identified score is incorrectly dropped.

    Fix: when the incoming item is unidentified, compare new_basic_score vs old_basic_score.
    - has_identify_scroll=True : scroll present; agent identifies and re-evaluates;
      identified score wins -> new sword equips, old dropped.  test_04 FAILS with
      current code because the sword is dropped before identification is attempted.
    - has_identify_scroll=False: no scroll; cannot identify -> new sword dropped in both
      old and new code (sanity check only, passes with current code)."""

    _MAGIC_SWORD_SEED = 0xB0000001   # identified magic 1H  (equips in test_02)
    _SCROLL_SEED      = 0xB0000002   # identify scroll       (gifted in test_03 when applicable)
    _UNID_SWORD_SEED  = 0xB0000003   # unidentified magic 1H (subject of the bug)

    def __init__(self, game, has_identify_scroll=True):
        self.game               = game
        self.has_identify_scroll = has_identify_scroll
        stand                   = _StandModel()
        runners                 = {lvl: stand for lvl in range(1, 17)}
        self.agent              = AgentAI(game, model_runners=runners, log=sys.stdout,
                                          use_two_hand_weapon=True)
        self.agent._test_mode   = True
        self._HL      = None
        self._HR      = None
        self._ss_seed = None   # starting sword seed (destroyed in test_02)
        self._bk_seed = None   # buckler seed (stays throughout)

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_1h_identified_magic(self, seed, name, mindam, maxdam, pldam_mod):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=1,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              pldam_mod=pldam_mod)

    def _gift_1h_unidentified_magic(self, seed, name, mindam, maxdam, pldam_mod):
        _gift(self.game, seed,
              itype=dx.ItemType.Sword.value,
              iloc=dx.item_equip_type.ILOC_ONEHAND.value,
              iclass=dx.item_class.ICLASS_WEAPON.value,
              icurs=_ICURS_LONG_SWORD,
              name=name, mindam=mindam, maxdam=maxdam,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              pldam_mod=pldam_mod)

    def _gift_scroll_of_identify(self, seed):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_SCROLL_OF,
              name="Scroll of Identify",
              mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_SCROLL.value,
              ididx=_IDI_IDENTIFY,
              ispell=dx.SpellID.Identify.value)

    def test_01_initial_state(self):
        """Verify clean start: short sword + buckler."""
        self._HL = dx.inv_item.INVITEM_HAND_LEFT.value
        self._HR = dx.inv_item.INVITEM_HAND_RIGHT.value
        self._settle()
        s = self._state()
        assert self._HL in s.body, "HAND_LEFT should have a weapon"
        assert self._HR in s.body, "HAND_RIGHT should have a shield"
        self._ss_seed = s.body[self._HL]
        self._bk_seed = s.body[self._HR]
        print(f"  ss={self._ss_seed:#010x}  bk={self._bk_seed:#010x}")

    def test_02_identified_magic_sword_equips(self):
        """Identified magic sword (basic_avg=10.0, identified_avg=15.0) beats starting
        short sword.  Agent uses full identified score -> equips, short sword destroyed."""
        HL = self._HL
        HR = self._HR
        # basic_avg = (8+12)/2 = 10.0; identified_avg = 10.0 + 5 = 15.0
        self._gift_1h_identified_magic(
            self._MAGIC_SWORD_SEED, "Magic Sword", mindam=8, maxdam=12, pldam_mod=5)
        self._settle()
        s = self._state()
        s.assert_body(HL, self._MAGIC_SWORD_SEED, "magic sword equipped")
        s.assert_body(HR, self._bk_seed,           "buckler unchanged")
        s.assert_absent(self._ss_seed,             "starting sword destroyed")

    def test_03_scroll_gifted_or_skipped(self):
        """If has_identify_scroll: gift scroll.  Equipped sword is already identified so
        _try_identify_with_new_scroll finds nothing to do; scroll kept in inventory.
        If not has_identify_scroll: nothing gifted (no scroll available for test_04)."""
        if not self.has_identify_scroll:
            return
        self._gift_scroll_of_identify(self._SCROLL_SEED)
        self._settle()
        s = self._state()
        s.assert_present(self._SCROLL_SEED, "scroll kept (nothing to identify)")
        s.assert_body(self._HL, self._MAGIC_SWORD_SEED, "magic sword still equipped")

    def test_04_unidentified_sword_decision(self):
        """Gift unidentified magic sword: basic_avg=16.0, pldam_mod=+3 (identified_avg=19.0).
        Scores against currently equipped identified magic sword (basic=10, identified=15, eff=187):

          new_basic_eff(16)=200 > old_eff(187)   -> new wins (basic eff beats identified eff)
          new_basic_eff(14)=175 < old_eff(187)   -> new LOSES (would fail with lower damage)
          new_identified_eff(19)=236 > old_eff   -> new wins after identification

        has_identify_scroll=True:
          - basic eff passes (200 > 187); identifies (236 > 187), new sword equips, old dropped.
        has_identify_scroll=False (passes with current code):
          - basic eff passes but no scroll -> new sword dropped."""
        HL = self._HL
        HR = self._HR
        # basic_avg = (14+18)/2 = 16.0; identified_avg = 16 + 3 = 19.0
        self._gift_1h_unidentified_magic(
            self._UNID_SWORD_SEED, "Better Unid Sword", mindam=14, maxdam=18, pldam_mod=3)
        self._settle(max_ticks=30)
        s = self._state()
        if self.has_identify_scroll:
            # FIX: identified (17 > 15) -> new sword equips, old dropped.
            # BUG: new sword dropped immediately (14 < 15); this assertion fails.
            s.assert_body(HL, self._UNID_SWORD_SEED,  "new sword equipped after identification")
            s.assert_body(HR, self._bk_seed,           "buckler unchanged")
            s.assert_absent(self._MAGIC_SWORD_SEED,   "old magic sword dropped")
            s.assert_absent(self._SCROLL_SEED,        "scroll consumed by identification")
        else:
            # No scroll: new sword dropped regardless (cannot identify to compare properly).
            s.assert_body(HL, self._MAGIC_SWORD_SEED, "magic sword still equipped")
            s.assert_body(HR, self._bk_seed,           "buckler unchanged")
            s.assert_absent(self._UNID_SWORD_SEED,    "unidentified sword dropped (no scroll)")

    def run_all(self):
        label = ("InvUnidentifiedUpgradeTests "
                 f"({'with' if self.has_identify_scroll else 'without'} scroll)")
        print(f"\n--- {label} ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


class InvJewelryTests:
    """Unidentified jewelry: equip into empty body slot without a scroll,
    drop when no slot is available.  After identification: keep items that
    score >= 0, drop cursed items (score < 0)."""

    _RING_SEED_1  = 0xEEFF0001
    _RING_SEED_2  = 0xEEFF0002
    _RING_SEED_3  = 0xEEFF0003
    _AMULET_SEED  = 0xEEFF0004
    _AMULET_SEED2 = 0xEEFF0005
    _SCROLL_SEED1 = 0xEEFF0011
    _SCROLL_SEED2 = 0xEEFF0012
    _SCROLL_SEED3 = 0xEEFF0013

    def __init__(self, game):
        self.game  = game
        stand      = _StandModel()
        runners    = {lvl: stand for lvl in range(1, 17)}
        self.agent = AgentAI(game, model_runners=runners, log=sys.stdout, use_two_hand_weapon=True)
        self.agent._test_mode = True
        self._RL = None   # INVITEM_RING_LEFT cii
        self._RR = None   # INVITEM_RING_RIGHT cii
        self._AM = None   # INVITEM_AMULET cii

    def _settle(self, max_ticks=_SETTLE_MAX_TICKS):
        _settle(self.agent, max_ticks)

    def _state(self):
        return _FullState(self.game.safe_state.player)

    def _gift_ring(self, seed, name, cursed=False):
        _gift(self.game, seed,
              itype=dx.ItemType.Ring.value,
              iloc=dx.item_equip_type.ILOC_RING.value,
              iclass=dx.item_class.ICLASS_MISC.value,
              icurs=_ICURS_RING,
              name=name, mindam=0, maxdam=0,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              plhp=-1000 if cursed else 0)

    def _gift_amulet(self, seed, name, cursed=False):
        _gift(self.game, seed,
              itype=dx.ItemType.Amulet.value,
              iloc=dx.item_equip_type.ILOC_AMULET.value,
              iclass=dx.item_class.ICLASS_MISC.value,
              icurs=_ICURS_AMULET,
              name=name, mindam=0, maxdam=0,
              identified=0,
              imagical=dx.item_quality.ITEM_QUALITY_MAGIC.value,
              plhp=-1000 if cursed else 0)

    def _gift_scroll_of_identify(self, seed):
        _gift(self.game, seed,
              itype=dx.ItemType.Misc.value,
              iloc=dx.item_equip_type.ILOC_NONE.value,
              iclass=dx.item_class.ICLASS_NONE.value,
              icurs=_ICURS_SCROLL_OF,
              name="Scroll of Identify",
              mindam=0, maxdam=0,
              imiscid=dx.item_misc_id.IMISC_SCROLL.value,
              ididx=_IDI_IDENTIFY,
              ispell=dx.SpellID.Identify.value)

    def test_01_initial_state(self):
        """Verify clean start: weapon + shield, all jewelry slots empty."""
        self._RL = dx.inv_item.INVITEM_RING_LEFT.value
        self._RR = dx.inv_item.INVITEM_RING_RIGHT.value
        self._AM = dx.inv_item.INVITEM_AMULET.value
        self._settle()
        s = self._state()
        assert self._RL not in s.body, "RING_LEFT should be empty initially"
        assert self._RR not in s.body, "RING_RIGHT should be empty initially"
        assert self._AM not in s.body, "AMULET should be empty initially"

    def test_02_ring1_equips_ring_left(self):
        """Unidentified ring, no scroll: occupies RING_LEFT to wait for scroll."""
        self._gift_ring(self._RING_SEED_1, "Ring 1")
        self._settle()
        s = self._state()
        s.assert_body(self._RL, self._RING_SEED_1, "ring1 in RING_LEFT")
        assert self._RR not in s.body, "RING_RIGHT still empty"

    def test_03_ring2_equips_ring_right(self):
        """Second unidentified ring, no scroll: occupies RING_RIGHT."""
        self._gift_ring(self._RING_SEED_2, "Ring 2")
        self._settle()
        s = self._state()
        s.assert_body(self._RL, self._RING_SEED_1, "ring1 still in RING_LEFT")
        s.assert_body(self._RR, self._RING_SEED_2, "ring2 in RING_RIGHT")

    def test_04_ring3_no_space_dropped(self):
        """Third unidentified ring, both ring slots taken: dropped."""
        self._gift_ring(self._RING_SEED_3, "Ring 3")
        self._settle()
        s = self._state()
        s.assert_body(self._RL, self._RING_SEED_1, "ring1 still in RING_LEFT")
        s.assert_body(self._RR, self._RING_SEED_2, "ring2 still in RING_RIGHT")
        s.assert_absent(self._RING_SEED_3, "ring3 dropped - no ring slot available")

    def test_05_cursed_amulet_equips(self):
        """Unidentified cursed amulet, no scroll: occupies AMULET slot
        (agent cannot know it is cursed until identified)."""
        self._gift_amulet(self._AMULET_SEED, "Cursed Amulet", cursed=True)
        self._settle()
        s = self._state()
        s.assert_body(self._AM, self._AMULET_SEED, "cursed amulet in AMULET slot")

    def test_06_amulet2_no_space_dropped(self):
        """Second unidentified amulet, AMULET slot already taken: dropped."""
        self._gift_amulet(self._AMULET_SEED2, "Amulet 2")
        self._settle()
        s = self._state()
        s.assert_body(self._AM, self._AMULET_SEED, "cursed amulet still in slot")
        s.assert_absent(self._AMULET_SEED2, "second amulet dropped - no amulet slot")

    def test_07_scrolls_rings_stay_cursed_amulet_dropped(self):
        """Three identify scrolls arrive: agent identifies all equipped jewelry.
        Rings score >= 0 after identification: stay.
        Cursed amulet scores < 0: dropped from body."""
        self._gift_scroll_of_identify(self._SCROLL_SEED1)
        self._gift_scroll_of_identify(self._SCROLL_SEED2)
        self._gift_scroll_of_identify(self._SCROLL_SEED3)
        self._settle(max_ticks=30)
        s = self._state()
        s.assert_body(self._RL, self._RING_SEED_1, "ring1 stays after identification")
        s.assert_body(self._RR, self._RING_SEED_2, "ring2 stays after identification")
        s.assert_absent(self._AMULET_SEED, "cursed amulet dropped after identification")

    def run_all(self):
        print(f"\n--- InvJewelryTests ---")
        tests = [m for m in dir(self) if m.startswith('test_')]
        passed = 0
        for name in sorted(tests):
            try:
                print(f"\n[RUN] {name}")
                getattr(self, name)()
                print(f"[OK]  {name}")
                passed += 1
            except Exception as e:
                import traceback
                tag = "FAIL" if isinstance(e, AssertionError) else "ERR"
                print(f"[{tag}] {name}: {e}")
                traceback.print_exc()
                print(f"\n{passed} passed, 1 failed (stopped)")
                return passed, 1
        print(f"\n{passed} passed, 0 failed")
        return passed, 0


def _load_config():
    ini = configparser.ConfigParser()
    ini.read('diablo-ai.ini')
    if 'default' not in ini:
        print("Error: diablo-ai.ini not found; run from the ai/ directory", file=sys.stderr)
        sys.exit(1)
    build_path = Path(ini['default']['diablo-build-path']).resolve()
    mshared    = ini['default']['diablo-mshared-filename']
    binary     = str(build_path / "devilutionx")
    if not (build_path / "spawn.mpq").exists():
        print(f"Error: spawn.mpq missing from {build_path}", file=sys.stderr)
        sys.exit(1)
    return binary, mshared


def _start_game(binary, mshared):
    config = {
        "mshared-filename":  mshared,
        "diablo-bin-path":   binary,
        "index":             0,
        "seed":              0,
        "fixed-seed":        True,
        "invincible-player": True,
        "no-monsters":       True,
        "blind-monsters":    False,
        "harmless-barrels":  True,
        "no_butcher":        True,
        "no-quests":         True,
        "spell-potency":     0.0,
        "no-spells":         False,
        "stats-scale":       1.0,
        "char-tables":       None,
        "hero-hp-min-pct":   100,
        "hero-hp-max-pct":   100,
        "hero-mana-min-pct": 100,
        "hero-mana-max-pct": 100,
        "hero-potions-min":  2,
        "hero-potions-max":  20,
        "no-auto-walk-on-seconday-action": True,
        "no-primary-action-doors":         False,
        "game-ticks-per-step": 1,
        "step-mode":           True,
        "gui":                 False,
        "dungeon-level":       [(1, 1)],
    }
    return diablo_state.DiabloGame.run(config)


def main():
    argparse.ArgumentParser(
        description="AgentAI inventory management integration tests"
    ).parse_args()

    binary, mshared = _load_config()

    # Generate devilutionx.py from the binary's DWARF info, then import.
    import devilutionx_generator
    devilutionx_generator.generate(binary)

    global dx, ring, diablo_state, AgentAI
    import devilutionx as dx
    import ring as ring
    import diablo_state as diablo_state
    from diablo_agent import AgentAI as _AgentAI
    AgentAI = _AgentAI

    game = _start_game(binary, mshared)

    suits = [
        InvTests,
        InvDurabilityTests,
        Inv2HBackupTests,
        Inv2HStaleBackupTests,
        lambda g: InvIdentifyTests(g, cursed=False),
        lambda g: InvIdentifyTests(g, cursed=True),
        lambda g: Inv2HIdentifyTests(g, cursed=False),
        lambda g: Inv2HIdentifyTests(g, cursed=True),
        Inv2HTo2HInheritanceTests,
        Inv2HTo1HInheritanceTests,
        Inv2HToShInheritanceTests,
        lambda g: Inv1HTo2HInheritanceTests(g, identified=False),
        lambda g: Inv1HTo2HInheritanceTests(g, identified=True),
        Inv1HTo1HInheritanceTests,
        lambda g: Inv1HShTo2HInheritanceTests(g, identified=False),
        lambda g: Inv1HShTo2HInheritanceTests(g, identified=True),
        Inv1HShToShInheritanceTests,
        InvBeatBackupSpamTests,
        Inv2HOscillationTests,
        lambda g: InvUnidentifiedUpgradeTests(g, has_identify_scroll=False),
        lambda g: InvUnidentifiedUpgradeTests(g, has_identify_scroll=True),
        InvJewelryTests,
    ]

    total_passed = total_failed = 0
    try:
        for suit_cl in suits:
            _reset_game(game)
            suite = suit_cl(game)
            p, f  = suite.run_all()
            total_passed += p
            total_failed += f
            if f:
                break
    finally:
        game.stop_or_detach()

    print(f"\n=== Total: {total_passed} passed, {total_failed} failed ===")
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()

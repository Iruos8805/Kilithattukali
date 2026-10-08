"""Headless smoke tests for game loop and modes."""

from kilithattu_game.app import create_game


def test_headless_smoke_play_crosser():
    game = create_game(mode="play_crosser", rules_preset="classic", headless=True)
    res = game.run(max_ticks=30)
    assert "score_attack" in res
    assert res["ticks"] >= 30


def test_headless_smoke_play_kili():
    game = create_game(mode="play_kili", rules_preset="classic", headless=True)
    res = game.run(max_ticks=30)
    assert "score_attack" in res
    assert res["ticks"] >= 30


def test_headless_smoke_watch_ai():
    game = create_game(mode="watch", rules_preset="classic", headless=True)
    res = game.run(max_ticks=30)
    assert "score_attack" in res
    assert res["ticks"] >= 30

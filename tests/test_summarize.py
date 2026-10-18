from lean_muscle.summarize import sentence_problem


def test_sentence_accepts_one_line_with_the_ratio():
    assert sentence_problem("肩腰 1.28，形状有了，腹肌这条是灯光在帮忙。", "1.28") is None


def test_hidden_arms_must_be_named():
    text = "肩腰比1.55，手臂都不漏，可能是太细了。"
    assert sentence_problem(text, "1.55", arms_hidden=True) is None
    assert "手臂都不漏" in sentence_problem("肩腰比1.55，胸肌还行。", "1.55", arms_hidden=True)
    assert "太细" in sentence_problem("肩腰比1.55，手臂都不漏。", "1.55", arms_hidden=True)
    assert sentence_problem(text, "1.55", arms_hidden=False) is not None


def test_sentence_rejects_a_rewritten_ratio_and_a_second_sentence():
    assert "1.28" in sentence_problem("肩腰 1.3，差不多。", "1.28")
    assert sentence_problem("肩腰 1.28，有了。再练。", "1.28") == "写成了两句"
    long = "肩腰 1.28，" + "很" * 40
    assert "超过" in sentence_problem(long, "1.28")
    assert "体脂" in sentence_problem("肩腰 1.28，体脂看着不低。", "1.28")

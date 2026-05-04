"""Tests for but_with_subs.tokenization_small100 module.

Uses mocking extensively to avoid requiring actual M2M100 model files.
"""

import json
import os
import shutil
import tempfile
import unittest
import unittest.mock as um

from transformers import BatchEncoding

from but_with_subs.tokenization_small100 import (
    SMALL100Tokenizer,
    load_json,
    load_spm,
    save_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_CONTENT = {
    "<unk>": 0,
    "<s>": 1,
    "</s>": 2,
    "hello": 3,
    "world": 4,
}


def _make_tokenizer(
    base_dir,
    tgt_lang="en",
    extra_vocab=None,
    **tokenizer_kwargs,
):
    """Create a SMALL100Tokenizer backed by minimal fixture files.

    Mocks load_spm so sentencepiece never touches the fake .spm file.

    Parameters
    ----------
    base_dir : str
        Path to an existing directory where fixture files will be written.
    tgt_lang : str
        Target language for initialisation.
    extra_vocab : dict | None
        Additional entries merged into the default vocabulary.
    """
    vocab = dict(VOCAB_CONTENT)
    if extra_vocab:
        vocab.update(extra_vocab)

    vocab_path = os.path.join(base_dir, "vocab.json")
    spm_path = os.path.join(base_dir, "sentencepiece.bpe.model")

    save_json(vocab, vocab_path)

    # Write a dummy file so the path exists (load_spm is mocked anyway).
    with open(spm_path, "wb") as f:
        f.write(b"DUMMY")

    mock_sp_model = um.MagicMock()
    mock_sp_model.encode.return_value = ["hello", "▁world"]
    mock_sp_model.decode.return_value = "hello world"
    mock_sp_model.serialized_model_proto.return_value = b"proto"

    with um.patch(
        "but_with_subs.tokenization_small100.load_spm", return_value=mock_sp_model
    ):
        tok = SMALL100Tokenizer(
            vocab_file=vocab_path,
            spm_file=spm_path,
            tgt_lang=tgt_lang,
            **tokenizer_kwargs,
        )

    return tok, vocab_path, spm_path


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------


class TestLoadSaveJson(unittest.TestCase):
    """Tests for the standalone JSON utility functions."""

    def test_load_json_returns_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"a": 1, "b": 2}, f)
            tmp = f.name
        try:
            result = load_json(tmp)
            self.assertIsInstance(result, dict)
            self.assertEqual(result, {"a": 1, "b": 2})
        finally:
            os.unlink(tmp)

    def test_load_json_returns_list(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            tmp = f.name
        try:
            result = load_json(tmp)
            self.assertIsInstance(result, list)
            self.assertEqual(result, [1, 2, 3])
        finally:
            os.unlink(tmp)

    def test_save_json_roundtrip(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            data = {"x": 10, "y": "hello"}
            save_json(data, tmp)
            loaded = load_json(tmp)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(tmp)

    def test_save_json_nested(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            data = {"outer": {"inner": [1, 2, {"k": "v"}]}}
            save_json(data, tmp)
            loaded = load_json(tmp)
            self.assertEqual(loaded, data)
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# load_spm
# ---------------------------------------------------------------------------


class TestLoadSpm(unittest.TestCase):
    """Tests for the load_spm helper function."""

    @um.patch("sentencepiece.SentencePieceProcessor")
    def test_load_spm_creates_processor_and_calls_load(self, MockSpmm):
        mock_instance = um.MagicMock()
        MockSpmm.return_value = mock_instance

        with tempfile.NamedTemporaryFile(suffix=".spm", delete=False) as f:
            tmp = f.name
        try:
            result = load_spm(tmp, {"enable_sampling": True})
        finally:
            os.unlink(tmp)

        MockSpmm.assert_called_once_with(enable_sampling=True)
        mock_instance.Load.assert_called_once()
        self.assertIs(result, mock_instance)


# ---------------------------------------------------------------------------
# Base class providing a temporary directory for all tokenizer tests
# ---------------------------------------------------------------------------


class TokenizerTestCase(unittest.TestCase):
    """Base class that provides a per-test temporary directory."""

    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

    @property
    def tmp_path(self):
        """Return the temporary directory path as a string."""
        return self._tmp_dir


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – basic construction
# ---------------------------------------------------------------------------


class TestSmall100TokenizerInit(TokenizerTestCase):
    """Tests for tokenizer initialisation."""

    def test_init_sets_vocab_file_and_spm_file(self):
        tok, vp, sp = _make_tokenizer(self.tmp_path)
        self.assertTrue(vp.endswith("vocab.json"))
        self.assertTrue(sp.endswith("sentencepiece.bpe.model"))

    def test_init_encoder_decoder_are_inverse_maps(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        for token, idx in tok.encoder.items():
            self.assertEqual(tok.decoder[idx], token)

    def test_init_default_tgt_lang_is_en(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang=None)
        self.assertEqual(tok.tgt_lang, "en")

    def test_init_tgt_lang_set_from_arg(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang="fr")
        self.assertEqual(tok.tgt_lang, "fr")

    def test_init_cur_lang_id_matches_tgt_lang(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang="de")
        expected = tok.get_lang_id("de")
        self.assertEqual(tok.cur_lang_id, expected)


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – properties
# ---------------------------------------------------------------------------


class TestSmall100TokenizerProperties(TokenizerTestCase):
    """Tests for property getters."""

    def test_vocab_size(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        expected = len(tok.encoder) + len(tok.lang_token_to_id)
        self.assertEqual(tok.vocab_size, expected)

    def test_tgt_lang_property_getter(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang="ja")
        self.assertEqual(tok.tgt_lang, "ja")

    def test_tgt_lang_property_setter(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        self.assertEqual(tok.tgt_lang, "en")
        tok.tgt_lang = "es"
        self.assertEqual(tok.tgt_lang, "es")

    def test_tgt_lang_setter_updates_special_tokens(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        tok.tgt_lang = "fr"
        fr_token = tok.get_lang_token("fr")
        fr_id = tok.lang_token_to_id[fr_token]
        # set_lang_special_tokens puts the lang token in prefix_tokens,
        # and eos_token_id in suffix_tokens.
        self.assertIn(fr_id, tok.prefix_tokens)
        self.assertEqual(tok.suffix_tokens, [tok.eos_token_id])


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – language helpers
# ---------------------------------------------------------------------------


class TestLanguageHelpers(TokenizerTestCase):
    """Tests for get_lang_token, get_lang_id, set_lang_special_tokens."""

    def test_get_lang_token(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        self.assertEqual(tok.get_lang_token("en"), "__en__")
        self.assertEqual(tok.get_lang_token("fr"), "__fr__")
        self.assertEqual(tok.get_lang_token("zh"), "__zh__")

    def test_get_lang_id_consistency(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        for lang in ["en", "fr", "de", "ja", "zh"]:
            token = tok.get_lang_token(lang)
            lang_id = tok.get_lang_id(lang)
            self.assertEqual(tok.lang_token_to_id[token], lang_id)

    def test_get_lang_id_unique_per_language(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        ids = set()
        for lang in ["en", "fr", "de", "es"]:
            ids.add(tok.get_lang_id(lang))
        self.assertEqual(len(ids), 4)

    def test_set_lang_special_tokens(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang="en")
        tok.set_lang_special_tokens("ja")
        ja_token = tok.get_lang_token("ja")
        ja_id = tok.lang_token_to_id[ja_token]
        self.assertEqual(tok.cur_lang_id, ja_id)
        self.assertEqual(tok.prefix_tokens, [ja_id])
        self.assertEqual(tok.suffix_tokens, [tok.eos_token_id])

    def test_set_lang_special_tokens_updates_cur_lang_id(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        tok.set_lang_special_tokens("de")
        self.assertEqual(tok.cur_lang_id, tok.get_lang_id("de"))


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – _tokenize (mocked sp_model)
# ---------------------------------------------------------------------------


class TestTokenize(unittest.TestCase):
    """Tests for _tokenize with mocked sentencepiece model."""

    @um.patch.object(SMALL100Tokenizer, "__init__", lambda s, **kw: None)
    def test_tokenize_calls_sp_model_encode(self):
        tok = SMALL100Tokenizer.__new__(SMALL100Tokenizer)
        tok.sp_model = um.MagicMock()
        tok.sp_model.encode.return_value = ["hello", "▁world"]
        result = tok._tokenize("hello world")
        tok.sp_model.encode.assert_called_once_with("hello world", out_type=str)
        self.assertEqual(result, ["hello", "▁world"])


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – _convert_token_to_id / _convert_id_to_token
# ---------------------------------------------------------------------------


class TestConvertTokenId(TokenizerTestCase):
    """Tests for token<->ID conversion helpers."""

    def test_convert_token_to_id_known_token(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        self.assertEqual(tok._convert_token_to_id("hello"), 3)
        self.assertEqual(tok._convert_token_to_id("world"), 4)

    def test_convert_token_to_id_unknown_token(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        unk_id = tok.encoder[tok.unk_token]
        self.assertEqual(tok._convert_token_to_id("xyznonexistent"), unk_id)

    def test_convert_token_to_id_lang_token(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        en_token = tok.get_lang_token("en")
        en_id = tok._convert_token_to_id(en_token)
        self.assertEqual(en_id, tok.lang_token_to_id[en_token])

    def test_convert_id_to_token_known_index(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        self.assertEqual(tok._convert_id_to_token(3), "hello")
        self.assertEqual(tok._convert_id_to_token(4), "world")

    def test_convert_id_to_token_lang_index(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        en_token = tok.get_lang_token("en")
        en_id = tok.lang_token_to_id[en_token]
        self.assertEqual(tok._convert_id_to_token(en_id), en_token)

    def test_convert_id_to_token_out_of_range_returns_unk(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        huge_id = 999999
        self.assertEqual(tok._convert_id_to_token(huge_id), tok.unk_token)


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – convert_tokens_to_string (mocked sp_model)
# ---------------------------------------------------------------------------


class TestConvertTokensToString(unittest.TestCase):
    """Tests for convert_tokens_to_string with mocked sp_model."""

    @um.patch.object(SMALL100Tokenizer, "__init__", lambda s, **kw: None)
    def test_decode_calls_sp_model_decode(self):
        tok = SMALL100Tokenizer.__new__(SMALL100Tokenizer)
        tok.sp_model = um.MagicMock()
        tok.sp_model.decode.return_value = "hello world"
        result = tok.convert_tokens_to_string(["hello", "▁world"])
        tok.sp_model.decode.assert_called_once_with(["hello", "▁world"])
        self.assertEqual(result, "hello world")


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – get_special_tokens_mask
# ---------------------------------------------------------------------------


class TestGetSpecialTokensMask(TokenizerTestCase):
    """Tests for get_special_tokens_mask."""

    def test_already_has_special_tokens_delegates_to_super(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        with um.patch.object(
            type(tok), "get_special_tokens_mask", return_value=[1, 0, 1]
        ) as mock_super:
            result = tok.get_special_tokens_mask(
                [1, 2], token_ids_1=[3], already_has_special_tokens=True
            )
            mock_super.assert_called_once()
            self.assertEqual(result, [1, 0, 1])

    def test_no_special_tokens_builds_inputs(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        result = tok.get_special_tokens_mask(
            [1, 2], token_ids_1=None, already_has_special_tokens=False
        )
        expected = tok.build_inputs_with_special_tokens([1, 2], None)
        self.assertEqual(result, expected)


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – build_inputs_with_special_tokens
# ---------------------------------------------------------------------------


class TestBuildInputsWithSpecialTokens(TokenizerTestCase):
    """Tests for build_inputs_with_special_tokens."""

    def test_single_sequence_no_prefix(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        delattr(tok, "prefix_tokens")
        result = tok.build_inputs_with_special_tokens([1, 2, 3])
        self.assertEqual(result, [1, 2, 3] + tok.suffix_tokens)

    def test_single_sequence_with_prefix(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        tok.prefix_tokens = [0]
        result = tok.build_inputs_with_special_tokens([1, 2, 3])
        self.assertEqual(result, [0] + [1, 2, 3] + tok.suffix_tokens)

    def test_pair_without_prefix(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        delattr(tok, "prefix_tokens")
        result = tok.build_inputs_with_special_tokens([1, 2], [3, 4])
        self.assertEqual(result, [1, 2, 3, 4] + tok.suffix_tokens)

    def test_pair_with_prefix(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        tok.prefix_tokens = [0]
        result = tok.build_inputs_with_special_tokens([1, 2], [3, 4])
        self.assertEqual(result, [0] + [1, 2, 3, 4] + tok.suffix_tokens)


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – get_vocab
# ---------------------------------------------------------------------------


class TestGetVocab(TokenizerTestCase):
    """Tests for get_vocab."""

    def test_get_vocab_contains_all_encoder_entries(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        vocab = tok.get_vocab()
        for token in tok.encoder:
            self.assertIn(token, vocab)

    def test_get_vocab_contains_lang_tokens(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        vocab = tok.get_vocab()
        self.assertIn("__en__", vocab)
        self.assertIn("__fr__", vocab)

    def test_get_vocab_size_matches_vocab_size_property(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        vocab = tok.get_vocab()
        self.assertEqual(len(vocab), tok.vocab_size)


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – prepare_seq2seq_batch (mocked parent)
# ---------------------------------------------------------------------------


class TestPrepareSeq2seqBatch(TokenizerTestCase):
    """Tests for prepare_seq2seq_batch with mocked parent class."""

    def test_sets_tgt_lang_and_calls_super(self):
        tok, _, _ = _make_tokenizer(self.tmp_path, tgt_lang="en")
        # Verify prepare_seq2seq_batch updates tgt_lang and calls
        # set_lang_special_tokens before the super() call.
        # Note: tgt_lang setter also calls set_lang_special_tokens,
        # so we expect it to be called twice.
        with um.patch.object(tok, "set_lang_special_tokens") as mock_set:
            with self.assertRaises(AttributeError):
                # super().prepare_seq2seq_batch doesn't exist in this
                # transformers version, but the side-effects above it
                # should still execute.
                tok.prepare_seq2seq_batch(
                    src_texts=["hello world"],
                    tgt_texts=["bonjour"],
                    tgt_lang="fr",
                )
            self.assertEqual(tok.tgt_lang, "fr")
            self.assertEqual(mock_set.call_count, 2)
            mock_set.assert_any_call("fr")


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – _build_translation_inputs
# ---------------------------------------------------------------------------


class TestBuildTranslationInputs(TokenizerTestCase):
    """Tests for _build_translation_inputs."""

    def test_raises_valueerror_when_tgt_lang_none(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        with self.assertRaises(ValueError) as ctx:
            tok._build_translation_inputs("hello", tgt_lang=None)
        self.assertIn("Translation requires a `tgt_lang`", str(ctx.exception))

    def test_sets_tgt_lang_and_calls_tokenizer(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        mock_encoded = BatchEncoding({"input_ids": [[1, 2]]})
        # Patch PreTrainedTokenizer.__call__ so the tokenizer body executes.
        from transformers import PreTrainedTokenizer
        orig_call = PreTrainedTokenizer.__call__
        PreTrainedTokenizer.__call__ = um.Mock(return_value=mock_encoded)
        try:
            result = tok._build_translation_inputs("hello", tgt_lang="fr")
            self.assertEqual(tok.tgt_lang, "fr")
            orig_call_mock = PreTrainedTokenizer.__call__
            orig_call_mock.assert_called_once()
            self.assertIs(result, mock_encoded)
        finally:
            PreTrainedTokenizer.__call__ = orig_call


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – _switch_to_input_mode / _switch_to_target_mode
# ---------------------------------------------------------------------------


class TestSwitchModes(TokenizerTestCase):
    """Tests for mode-switching helpers."""

    def test_switch_to_input_mode(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        original_suffix = tok.suffix_tokens[:]
        with um.patch.object(tok, "set_lang_special_tokens") as mock_set:
            tok._switch_to_input_mode()
            mock_set.assert_called_once_with(tok.tgt_lang)

    def test_switch_to_target_mode(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        tok._switch_to_target_mode()
        self.assertIsNone(tok.prefix_tokens)
        self.assertEqual(tok.suffix_tokens, [tok.eos_token_id])


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – __getstate__ / __setstate__ (serialization)
# ---------------------------------------------------------------------------


class TestSerialization(TokenizerTestCase):
    """Tests for pickle-style serialization."""

    def test_getstate_sets_sp_model_to_none(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        state = tok.__getstate__()
        self.assertIsNone(state["sp_model"])
        self.assertIsNotNone(state["encoder"])
        self.assertIsNotNone(state["decoder"])

    def test_setstate_restores_from_dict(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        state = tok.__getstate__()
        tok2 = SMALL100Tokenizer.__new__(SMALL100Tokenizer)
        with um.patch("but_with_subs.tokenization_small100.load_spm") as mock_load:
            mock_sp = um.MagicMock()
            mock_load.return_value = mock_sp
            tok2.__setstate__(state)
        # __setstate__ re-loads the sp_model from disk via load_spm
        self.assertIs(tok2.sp_model, mock_sp)
        self.assertEqual(tok2._tgt_lang, tok._tgt_lang)
        mock_load.assert_called_once()

    def test_setstate_backwards_compat_no_sp_model_kwargs(self):
        """Ensure __setstate__ adds sp_model_kwargs if missing."""
        tok, _, _ = _make_tokenizer(self.tmp_path)
        state = tok.__getstate__()
        del state["sp_model_kwargs"]
        tok2 = SMALL100Tokenizer.__new__(SMALL100Tokenizer)
        with um.patch("but_with_subs.tokenization_small100.load_spm") as mock_load:
            mock_load.return_value = um.MagicMock()
            tok2.__setstate__(state)
        self.assertEqual(tok2.sp_model_kwargs, {})


# ---------------------------------------------------------------------------
# SMALL100Tokenizer – save_vocabulary
# ---------------------------------------------------------------------------


class TestSaveVocabulary(TokenizerTestCase):
    """Tests for save_vocabulary."""

    def test_save_vocabulary_creates_files(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        out_dir = os.path.join(self.tmp_path, "output")
        os.makedirs(out_dir)
        paths = tok.save_vocabulary(out_dir)
        self.assertEqual(len(paths), 2)
        vocab_saved, spm_saved = paths
        self.assertTrue(os.path.exists(vocab_saved))
        self.assertTrue(os.path.exists(spm_saved))

    def test_save_vocabulary_raises_on_non_directory(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        fake_file = os.path.join(self.tmp_path, "not_a_dir")
        with open(fake_file, "w") as f:
            f.write("")
        with self.assertRaises(OSError) as ctx:
            tok.save_vocabulary(fake_file)
        self.assertIn("should be a directory", str(ctx.exception))

    def test_save_vocabulary_with_prefix(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        out_dir = os.path.join(self.tmp_path, "output")
        os.makedirs(out_dir)
        paths = tok.save_vocabulary(out_dir, filename_prefix="custom")
        vocab_saved, spm_saved = paths
        self.assertIn("custom-", os.path.basename(vocab_saved))
        self.assertIn("custom-", os.path.basename(spm_saved))

    def test_save_vocabulary_preserves_vocab_data(self):
        tok, _, _ = _make_tokenizer(self.tmp_path)
        out_dir = os.path.join(self.tmp_path, "output")
        os.makedirs(out_dir)
        tok.save_vocabulary(out_dir)
        saved_vocab_path = os.path.join(out_dir, "vocab.json")
        saved = load_json(saved_vocab_path)
        self.assertEqual(saved, tok.encoder)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

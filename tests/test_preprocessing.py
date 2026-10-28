"""Passo 2 do TCC de PDI: as cinco tecnicas, o modo neutro e as combinacoes."""
import unittest

import numpy as np

from App.preprocessing import DEFAULT_PARAMS, Preprocessor, TECHNIQUES


def scene(seed=7):
    """Frame BGR com textura e pouco contraste, como uma cena escura."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 90, size=(48, 64, 3), dtype=np.uint8)


class PreprocessingTests(unittest.TestCase):
    def test_every_technique_keeps_format_and_is_deterministic(self):
        frame = scene()
        for name in TECHNIQUES:
            first = Preprocessor([(name, {})])(frame.copy())
            self.assertEqual((first.shape, first.dtype), (frame.shape, frame.dtype), name)
            np.testing.assert_array_equal(first, Preprocessor([(name, {})])(frame.copy()))

    def test_neutral_returns_the_frame_unchanged(self):
        frame = scene()
        self.assertIs(Preprocessor([])(frame), frame)
        self.assertIs(Preprocessor.from_spec("nenhuma")(frame), frame)

    def test_techniques_change_the_image(self):
        frame = scene()
        for name in ("equalizacao", "clahe", "gama", "gaussiano", "nitidez"):
            self.assertFalse(np.array_equal(Preprocessor([(name, {})])(frame.copy()), frame), name)

    def test_luminance_techniques_keep_gray_pixels_gray(self):
        # Cinza nao tem cor: mexer so na luminancia nao pode criar cor.
        gray = np.repeat(scene()[:, :, :1], 3, axis=2)
        for name in ("equalizacao", "clahe", "gama"):
            out = Preprocessor([(name, {})])(gray.copy()).astype(int)
            self.assertLessEqual(np.abs(out[:, :, 0] - out[:, :, 2]).max(), 1, name)

    def test_gamma_below_one_brightens_dark_pixels(self):
        frame = np.full((4, 4, 3), 64, dtype=np.uint8)
        out = Preprocessor([("gama", {"gamma": 0.5})])(frame)
        # 255 * (64 / 255) ** 0.5 = 127,7
        self.assertTrue(np.all(np.abs(out.astype(int) - 128) <= 1))

    def test_order_of_a_combination_is_kept(self):
        frame = scene()
        first = Preprocessor.from_spec("clahe+gaussiano")
        second = Preprocessor.from_spec("gaussiano+clahe")
        self.assertEqual([step["tecnica"] for step in first.config], ["clahe", "gaussiano"])
        self.assertFalse(np.array_equal(first(frame.copy()), second(frame.copy())))

    def test_spec_overrides_defaults_and_config_records_them(self):
        config = Preprocessor.from_spec("gama:gamma=0.6+nitidez:amount=1.0,sigma=2.0").config
        self.assertEqual(config, [{"tecnica": "gama", "gamma": 0.6},
                                  {"tecnica": "nitidez", "sigma": 2.0, "amount": 1.0}])
        self.assertEqual(Preprocessor.from_spec("clahe").config,
                         [{"tecnica": "clahe", **DEFAULT_PARAMS["clahe"]}])

    def test_unknown_technique_or_parameter_stops(self):
        with self.assertRaisesRegex(ValueError, "desconhecida"):
            Preprocessor.from_spec("retinex")
        with self.assertRaisesRegex(ValueError, "Parametros desconhecidos"):
            Preprocessor.from_spec("clahe:limite=3.0")


if __name__ == "__main__":
    unittest.main()

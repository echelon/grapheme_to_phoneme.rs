use arpabet::phoneme::Consonant;
use arpabet::phoneme::Phoneme;
use arpabet::phoneme::Vowel;
use arpabet::phoneme::VowelStress;
use crate::PhonemeToken;

#[allow(dead_code)]
pub (crate) fn build_phoneme_string_table() -> Vec<&'static str> {
  // TODO: Grounds for optimization with lazy_static! macro.
  vec![
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
    "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D",
    "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F",
    "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
    "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
    "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
    "Z", "ZH"
  ]
}

pub (crate) fn build_phoneme_token_arpabet_table() -> Vec<PhonemeToken> {
  // NB: This must match the order that the network was trained on, otherwise the results
  // will be corrupted.
  // TODO: Grounds for optimization with lazy_static! macro.
  vec![
    // TODO: arpabet.rs should be cleaned up. This is too awkward to construct and not reusable.
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AA(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AA(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AA(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AE(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AE(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AE(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AH(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AH(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AH(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AO(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AO(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AO(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AW(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AW(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AW(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AY(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AY(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::AY(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::B)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::CH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::D)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::DH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EH(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EH(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EH(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::ER(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::ER(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::ER(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EY(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EY(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::EY(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::F)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::G)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::HH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IH(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IH(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IH(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IY(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IY(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::IY(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::JH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::K)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::L)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::M)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::N)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::NG)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OW(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OW(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OW(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OY(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OY(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::OY(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::P)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::R)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::S)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::SH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::T)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::TH)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UH(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UH(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UH(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UW(VowelStress::UnknownStress))), // NB: Mistake in g2p.py?
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UW(VowelStress::NoStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UW(VowelStress::PrimaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Vowel(Vowel::UW(VowelStress::SecondaryStress))),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::V)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::W)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::Y)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::Z)),
    PhonemeToken::ArpabetPhoneme(Phoneme::Consonant(Consonant::ZH)),
  ]
}

#[cfg(test)]
mod tests {
  use crate::tables::{build_phoneme_string_table, build_phoneme_token_arpabet_table};
  use crate::PhonemeToken;
  use arpabet::phoneme::Phoneme;

  #[test]
  fn assert_expected_ordering() {
    // It's a bit dangerous to rebuild the table with arpabet.rs since it's so verbose
    // and hard to maintain. We check that the order is maintained as expected.

    // We treat this string table as authoritative.
    let string_table = build_phoneme_string_table();

    // Whereas this is subject to change.
    let token_table = build_phoneme_token_arpabet_table();

    assert_eq!(string_table.len(), token_table.len());

    let phonemes : Vec<Phoneme> = token_table.iter()
      .filter(|token| match token {
        PhonemeToken::ArpabetPhoneme(_) => true,
        _ => false,
      })
      .map(|token| match token {
        PhonemeToken::ArpabetPhoneme(inner) => inner.to_owned(),
        _ => unreachable!()
      })
      .collect();

    let phoneme_strings : Vec<&str> = phonemes.iter()
      .map(|p| p.to_str())
      .collect();

    for i in 0..token_table.len() {
      let expected = string_table.get(i).map(|c| *c).expect("length already asserted");
      let actual = phoneme_strings.get(i).map(|c| *c).expect("length already asserted");
      assert_eq!(expected, actual);
    }
  }
}

// Copyright (c) 2020 Brandon Thomas <bt@brand.io>, <echelon@gmail.com>

//#![deny(dead_code)]
//#![deny(missing_docs)]
//#![deny(unreachable_patterns)]
//#![deny(unused_extern_crates)]
//#![deny(unused_imports)]
//#![deny(unused_qualifications)]

//! **Phoneme** is a prediction tool to turn graphemes into phonemes.
//! It is based on g2p.py

use std::fs::File;
use std::io::Read;
use std::{io, fmt};
use std::error::Error;
use std::fmt::Formatter;
use ndarray::Slice;
use ndarray::{Array1, Axis, Array3, ArrayBase, Array0, ArrayView2};
use ndarray::Array2;
use ndarray_npy::{NpzReader, ReadNpzError, read_npy, ReadNpyError};
use std::collections::HashMap;
use ndarray::linalg::general_mat_mul;

pub struct Phoneme {
}


pub struct Model {
  graphemes: Vec<String>,
  phonemes: Vec<String>,
  enc_emb : Array2<f32>,
  enc_w_ih : Array2<f32>,
  enc_w_hh : Array2<f32>,
  enc_b_ih : Array1<f32>,
  enc_b_hh : Array1<f32>,

  dec_emb : Array2<f32>,
  dec_w_ih : Array2<f32>,
  dec_w_hh : Array2<f32>,
  dec_b_ih : Array1<f32>,
  dec_b_hh : Array1<f32>,

  fc_w : Array2<f32>,
  fc_b : Array1<f32>,

  g2idx: HashMap<String, usize>,
  idx2p: HashMap<usize, String>,

  unknown_grapheme_idx: usize,
}

impl Model {
  pub fn read_npy() -> Result<Self, PhonemeError> {
    let mut graphemes = Vec::new();
    graphemes.push("<pad>".to_string());
    graphemes.push("<unk>".to_string());
    graphemes.push("</s>".to_string());

    let characters : Vec<String> = "abcdefghijklmnopqrstuvwxyz".chars()
      .map(|c| c.to_string())
      .collect();

    graphemes.extend(characters);

    let phones = vec![
      "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
      "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D",
      "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1", "EY2", "F",
      "G", "HH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L", "M",
      "N", "NG", "OW0", "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH",
      "T", "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W", "Y",
      "Z", "ZH"
    ];

    let mut phonemes = Vec::new();
    phonemes.push("<pad>".to_string());
    phonemes.push("<unk>".to_string());
    phonemes.push("<s>".to_string());
    phonemes.push("</s>".to_string());
    phonemes.extend(phones.iter().map(|p| p.to_string()).collect::<Vec<String>>());

    let mut g2idx : HashMap<String, usize> = HashMap::new();
    for (i, val) in graphemes.iter().enumerate() {
      g2idx.insert(val.to_string(), i);
    }

    let unknown_grapheme_idx = g2idx.get("<unk>")
      .map(|u| u.clone())
      .expect("<unk> should be a grapheme");

    let mut idx2p : HashMap<usize, String> = HashMap::new();
    for (i, val) in phonemes.iter().enumerate() {
      idx2p.insert(i, val.to_string());
    }

    let enc_emb : Array2<f32> = read_npy("data/enc_emb.npy")?; // (29, 64). (len(graphemes), emb)
    let enc_w_ih : Array2<f32> = read_npy("data/enc_w_ih.npy")?; // (3*128, 64)
    let enc_w_hh : Array2<f32> = read_npy("data/enc_w_hh.npy")?; // (3*128, 128)
    let enc_b_ih : Array1<f32> = read_npy("data/enc_b_ih.npy")?; // (3*128,)
    let enc_b_hh : Array1<f32> = read_npy("data/enc_b_hh.npy")?; // (3*128,)

    let dec_emb : Array2<f32> = read_npy("data/dec_emb.npy")?; // (74, 64). (len(phonemes), emb)
    let dec_w_ih : Array2<f32> = read_npy("data/dec_w_ih.npy")?; // (3*128, 64)
    let dec_w_hh : Array2<f32> = read_npy("data/dec_w_hh.npy")?; // (3*128, 128)
    let dec_b_ih : Array1<f32> = read_npy("data/dec_b_ih.npy")?; // (3*128,)
    let dec_b_hh : Array1<f32> = read_npy("data/dec_b_hh.npy")?; // (3*128,)

    let fc_w : Array2<f32> = read_npy("data/fc_w.npy")?; // (74, 128)
    let fc_b : Array1<f32> = read_npy("data/fc_b.npy")?; // (74,)

    Ok(Self {
      graphemes,
      phonemes,
      enc_emb,
      enc_w_ih,
      enc_w_hh,
      enc_b_ih,
      enc_b_hh,
      dec_emb,
      dec_w_ih,
      dec_w_hh,
      dec_b_ih,
      dec_b_hh,
      fc_w,
      fc_b,
      g2idx,
      idx2p,
      unknown_grapheme_idx,
    })
  }

  pub fn predict(&self, grapheme: &str) -> Vec<String> {
    let enc = self.encode(grapheme);
    let enc = self.gru(&enc, grapheme.len() + 1);

    let last_hidden = enc.index_axis(Axis(1), grapheme.len()); // TODO: correct?

    let mut dec = self.dec_emb.index_axis(Axis(0), 2)// 2 = <s>
      .insert_axis(Axis(0)); // (1, 256)

    let mut h = last_hidden.to_owned();

    let mut preds = Vec::new();

    for i in 0..20 {
      h = self.grucell(&dec, &h); // TODO: Need to pass decoder, not encoder!

      // For 2d arrays, `dot(&Rhs)` computes the matrix multiplication
      let logits = h.dot(&self.fc_w.t()) + &self.fc_b;
      let pred = self.argmax(&logits);

      if pred == 3 {
        break; // 3 = </s>
      }

      preds.push(pred);

      dec = self.dec_emb.index_axis(Axis(0), pred)
        .insert_axis(Axis(0)); // (1, 256)
    }

    let preds = preds.iter()
      .map(|idx| self.idx2p.get(&idx)
        .map(|x| x.clone())
        .unwrap_or("<unk>".to_string()))
      .collect();

    preds
  }

  pub fn encode(&self, grapheme: &str) -> Array3<f32> {
    let mut chars : Vec<String> = grapheme.chars()
      .map(|c| c.to_string())
      .collect();

    chars.push("</s>".to_string());

    // Incredibly useful guide for porting NumPy to Rust ndarray:
    // https://docs.rs/ndarray/0.13.1/ndarray/doc/ndarray_for_numpy_users/index.html

    let encoded : Vec<usize> = chars.iter()
      .map(|c| c.to_string())
      .map(|c| self.g2idx.get(&c)
        .map(|u| u.clone())
        .unwrap_or(self.unknown_grapheme_idx))
      .collect();

    let shape = (encoded.len(), 256);
    let mut embeddings : Array2<f32> = Array2::zeros(shape);

    for (i, mut row) in embeddings.axis_iter_mut(Axis(0)).enumerate() {
      let embedding_index = *encoded.get(i).expect("error handling"); // TODO ERROR HANDLING
      let embedding = self.enc_emb.index_axis(Axis(0), embedding_index);
      row.assign(&embedding);
    }

    embeddings.insert_axis(Axis(0)) // (1, N, 256)
  }

  /*
    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    ----------

  enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
   */
  pub fn gru(&self, x: &Array3<f32>, steps: usize) -> Array3<f32> {
    // Initial hidden state
    // np.zeros((1, self.enc_w_hh.shape[-1]), np.float32)
    let mut h : Array2<f32> = Array2::zeros((1, 256));
    let mut outputs : Array3<f32> = Array3::zeros((1, steps, 256));

    /*for i in 0..steps {
      let sub_x = x.index_axis(Axis(1), 1);
      //println!("Subx shape: {:?}", sub_x.shape());
      h = self.grucell(&sub_x, &h);
    }*/

    for (_i, mut row) in outputs.axis_iter_mut(Axis(0)).enumerate() {
      let sub_x = x.index_axis(Axis(1), 1);
      //println!("Subx shape: {:?}", sub_x.shape());
      h = self.grucell(&sub_x, &h);
      row.assign(&h);
    }

    outputs // (1, N, 256)
  }

  /*

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h
   */
  /// x: (1, 256)
  /// h: (1, 256)
  pub fn grucell(&self, x: &ArrayView2<f32>, h: &Array2<f32>) -> Array2<f32> {
    //general_mat_mul()
    // For 2d arrays, `dot(&Rhs)` computes the matrix multiplication
    let rzn_ih  = x.dot(&self.enc_w_ih.t()) + &self.enc_b_ih;
    let rzn_hh  = h.dot(&self.enc_w_hh.t()) + &self.enc_b_hh;

    let t_ih = rzn_ih.shape()[1] * 2 / 3;
    let rz_ih = rzn_ih.slice_axis(Axis(1), Slice::from(0..t_ih));
    let n_ih = rzn_ih.slice_axis(Axis(1), Slice::from(t_ih..));

    let t_hh = rzn_hh.shape()[1] * 2 / 3;
    let rz_hh = rzn_hh.slice_axis(Axis(1), Slice::from(0..t_hh));
    let n_hh = rzn_hh.slice_axis(Axis(1), Slice::from(t_hh..));

    // TODO: Inefficient. Can't add views.
    let rz_ih : Array2<f32> = rz_ih.to_owned();
    let rz_hh : Array2<f32> = rz_hh.to_owned();

    let result = rz_ih + rz_hh;
    let rz = self.sigmoid(&result);

    let (r, z) = rz.view().split_at(Axis(1), 256); // TODO The math isn't working!

    let n_ih : Array2<f32> = n_ih.to_owned();
    let n_hh : Array2<f32> = n_hh.to_owned();

    let inner = n_ih + r + n_hh;
    let n = inner.map(|x: &f32| x.tanh());

    let z = z.into_owned();

    let h = (z.map(|x: &f32| 1.0 - x)) * n + z * h;

    h // output is (1, 256)
  }

  /// x: (1, 512)
  /// output: (1, 512)
  pub fn sigmoid(&self, x: &Array2<f32>) -> Array2<f32> {
    let x : Array2<f32> = x.map(|x: &f32| 1.0 / (1.0 + (-x).exp()));
    x
  }

  // TODO TEST AND DEBUG
  pub fn argmax(&self, x: &Array2<f32>) -> usize {
    let mut max : f32 = *x.get((0,0)).expect("todo"); // todo error handling
    let mut argmax = 0;

    let mut i = 0;
    for y in x.slice_axis(Axis(1), Slice::from(0..1)) {
      let y = *y;
      if y > max {
        max = y;
        argmax = i;
      }
      i += 1;
    }
    argmax
  }
}

impl Phoneme {
  pub fn new() -> Self {
    Self {}
  }

  /*pub fn grapheme_to_phoneme(&self, polyphone: &str) -> String {
    "".to_string()
  }*/
}

#[derive(Debug)]
pub enum PhonemeError {
  IoError(std::io::Error),
  ReadNpyError(ndarray_npy::ReadNpyError),
  ReadNpzError(ndarray_npy::ReadNpzError),
}

impl From<io::Error> for PhonemeError {
  fn from(e: io::Error) -> Self {
    PhonemeError::IoError(e)
  }
}

impl From<ReadNpzError> for PhonemeError {
  fn from(e: ReadNpzError) -> Self {
    PhonemeError::ReadNpzError(e)
  }
}

impl From<ReadNpyError> for PhonemeError {
  fn from(e: ReadNpyError) -> Self {
    PhonemeError::ReadNpyError(e)
  }
}

impl fmt::Display for PhonemeError {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    write!(f, "PhonemeError")
  }
}

impl Error for PhonemeError {
  fn source(&self) -> Option<&(dyn Error + 'static)> {
    match &self {
      PhonemeError::IoError(e) => Some(e),
      PhonemeError::ReadNpzError(e) => Some(e),
      PhonemeError::ReadNpyError(e) => Some(e),
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::Model;
  use anyhow::Result as AnyhowResult;

  #[test]
  fn test_read_npz() -> AnyhowResult<()> {
    let model = Model::read_npy()?;
    Ok(())
  }

  #[test]
  fn test_encode() -> AnyhowResult<()> {
    let model = Model::read_npy()?;
    let predicted = model.predict("test");

    println!("Predicted: {:?}", predicted);

    assert_eq!(1,2);

    Ok(())
  }
}

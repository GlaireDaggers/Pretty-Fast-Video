use std::io::{Read, Seek};

use bitstream_io::{BitRead, BitReader, Endianness};

#[derive(Debug)]
pub enum HuffmanError {
    DecodeError,
    IOError(std::io::Error),
}

pub struct HuffmanTree {
    codes: [Code;16],
    table: [u8;16],
    dec_table: [Code;256],
    root: Box<Node>,
}

#[derive(Clone, Copy)]
pub struct Code {
    pub val: u32,
    pub len: u32,
    pub symbol: u8,
}

impl Code {
    pub fn new() -> Code {
        Code { val: 0, len: 0, symbol: 0 }
    }

    pub fn append(self: &Code, bit: bool) -> Code {
        Code { val: self.val | ((bit as u32) << self.len), len: self.len + 1, symbol: self.symbol }
    }

    pub fn mask(self: &Code) -> u32 {
        (1 << self.len) - 1
    }
}

struct Node {
    freq: u32,
    ch: Option<u8>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    pub fn new(freq: u32, ch: Option<u8>) -> Node {
        Node { freq: freq, ch: ch, left: None, right: None }
    }

    pub fn into_box(self: Node) -> Box<Node> {
        Box::new(self)
    }
}

impl HuffmanTree {
    pub fn empty() -> HuffmanTree {
        HuffmanTree { codes: [Code::new();16], table: [0;16], dec_table: [Code::new();256], root: Node::new(0, None).into_box() }
    }

    fn get_insert_index(node: &Box<Node>, p: &[Box<Node>]) -> usize {
        for i in 0..p.len() {
            if node.freq > p[i].freq {
                return i;
            }
        }

        return p.len();
    }

    pub fn from_table(table: &[u8;16]) -> HuffmanTree {
        let mut p:Vec<Box<Node>> = Vec::new();

        for (ch, fr) in table.iter().enumerate() {
            if *fr > 0 {
                p.push(Node::new(*fr as u32, Some(ch as u8)).into_box());
            }
        }

        // start with a sorted list
        p.sort_by(|a, b| (&(b.freq)).cmp(&(a.freq)));

        while p.len() > 1 {
            let a = p.pop().unwrap();
            let b = p.pop().unwrap();
            let mut c = Node::new(a.freq + b.freq, None).into_box();
            c.left = Some(a);
            c.right = Some(b);

            // insertion sort new node back into list
            let insert_pos = HuffmanTree::get_insert_index(&c, &p);
            p.insert(insert_pos, c);
        }

        if p.len() == 0 {
            return HuffmanTree::empty();
        }

        let root = p.pop().unwrap();

        let mut codes = [Code::new();16];
        assign_codes(&root, &mut codes, Code::new());

        // generate pre-masked decoder table for codes of 8 bits or less - allows us to just read in a whole u8 and index into this table to get a code
        // if a code is longer than 8 bits, it falls back to the slow tree traversal path

        let mut dec_table = [Code::new();256];

        for val in 0..256 {
            for c in codes {
                if c.len > 0 && c.len <= 8 && val & c.mask() == c.val {
                    dec_table[val as usize] = c;
                    break;
                }
            }
        }

        HuffmanTree { codes: codes, table: table.clone(), dec_table: dec_table, root: root }
    }

    pub fn get_table(self: &HuffmanTree) -> &[u8;16] {
        &self.table
    }

    fn read_slow<R: Read, E: Endianness>(self: &HuffmanTree, bitreader: &mut BitReader<R, E>) -> Result<u8, HuffmanError> {
        let mut node = &self.root;

        loop {
            if let Some(ch) = node.ch {
                return Ok(ch);
            } else {
                let bit = match bitreader.read_bit() {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(HuffmanError::IOError(e));
                    }
                };

                if bit {
                    if let Some(ref r) = node.right {
                        node = r;
                    } else {
                        return Err(HuffmanError::DecodeError);
                    }
                } else {
                    if let Some(ref l) = node.left {
                        node = l;
                    } else {
                        return Err(HuffmanError::DecodeError);
                    }
                }
            }
        }
    }

    pub fn read<R: Read + Seek, E: Endianness>(self: &HuffmanTree, reader: &mut BitReader<R, E>, max_bits: u64) -> Result<u8, HuffmanError> {
        // workaround to avoid reading past the end of the stream
        let bit_pos = match reader.position_in_bits() {
            Ok(v) => v,
            Err(e) => {
                return Err(HuffmanError::IOError(e));
            }
        };
        let bits_remaining = max_bits - bit_pos;
        let read_bits = bits_remaining.min(8);

        let cur = match reader.read::<u8>(read_bits as u32) {
            Ok(v) => v,
            Err(e) => {
                return Err(HuffmanError::IOError(e));
            }
        };

        let c = self.dec_table[cur as usize];
        if c.len == 0 {
            // couldn't find code in fast table, try slow lookup instead
            match reader.seek_bits(std::io::SeekFrom::Current(-(read_bits as i64))) {
                Ok(v) => v,
                Err(e) => {
                    return Err(HuffmanError::IOError(e));
                }
            };

            return self.read_slow(reader);
        } else {
            let rewind = (read_bits as i64) - c.len as i64;
            
            match reader.seek_bits(std::io::SeekFrom::Current(-rewind)) {
                Ok(v) => v,
                Err(e) => {
                    return Err(HuffmanError::IOError(e));
                }
            };

            return Ok(c.symbol);
        }
    }

    pub fn get_code(self: &HuffmanTree, val: u8) -> Code {
        self.codes[val as usize]
    }
}

fn assign_codes(p: &Box<Node>, h: &mut [Code;16], s: Code) {
    if let Some(ch) = p.ch {
        let s = Code { val: s.val, len: s.len, symbol: ch };
        h[ch as usize] = s;
    } else {
        if let Some(ref l) = p.left {
            assign_codes(l, h, s.append(false));
        }

        if let Some(ref r) = p.right {
            assign_codes(r, h, s.append(true));
        }
    }
}
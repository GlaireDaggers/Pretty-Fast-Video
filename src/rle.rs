use crate::huffman::HuffmanTree;

pub struct RLESequence {
    pub num_zeroes: u8,
    pub coeff_size: u8,
    pub coeff: i16,
}

pub fn rle_encode(into: &mut Vec<RLESequence>, data: &[i16]) {
    let mut run: u32 = 0;

    for idx in 0..data.len() {
        let val = data[idx];

        if val == 0 {
            run += 1;
        } else {
            while run > 15 {
                into.push(RLESequence { num_zeroes: 15, coeff_size: 0, coeff: 0 });
                run -= 15;
            }

            let c = val.abs() as u16;
            let numbits = (16 - c.leading_zeros()) + 1;

            into.push(RLESequence { num_zeroes: run as u8, coeff_size: numbits as u8, coeff: val });
            run = 0;
        }
    }

    while run > 15 {
        into.push(RLESequence { num_zeroes: 15, coeff_size: 0, coeff: 0 });
        run -= 15;
    }

    if run > 0 {
        into.push(RLESequence { num_zeroes: run as u8, coeff_size: 0, coeff: 0 });
    }
}

pub fn update_table(table: &mut [i32;16], sequence: &[RLESequence]) {
    for s in sequence {
        debug_assert!(s.num_zeroes < 16 && s.coeff_size < 16);
        table[s.num_zeroes as usize] += 1;
        table[s.coeff_size as usize] += 1;
    }
}

pub fn rle_create_huffman(table: &[i32;16]) -> HuffmanTree {
    let mut max = 0;
    for x in table {
        max = max.max(*x);
    }

    let table = table.map(|x| {
        if x > 0 {
            let val = ((x * 255) / max).max(1) as u8;
            debug_assert!(val > 0);
            val
        } else {
            0
        }
    });

    HuffmanTree::from_table(&table)
}
//! Debug: dump PLE metadata tensor payloads through the real reader.
use colibri_format::package::Package;

fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let pkg = Package::open(std::path::Path::new(&dir)).expect("open");
    let targets = [
        "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
        "layers.1.ple.ple_embedding.ngram_heads_offsets",
        "layers.1.ple.ple_embedding.layer_multipliers",
    ];
    for t in targets {
        let rec = pkg.record_by_name(t).expect(t);
        let payload = pkg.read_tensor_payload(rec).expect("payload");
        println!("== {t}: stored={} decoded={} payload={}", rec.stored, rec.decoded, payload.len());
        let hex: Vec<String> = payload.iter().take(32).map(|b| format!("{b:02x}")).collect();
        println!("  head: {}", hex.join(" "));
        // try i32, f32, bf16
        let i32s: Vec<i32> = payload.chunks_exact(4).take(6).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        let f32s: Vec<f32> = payload.chunks_exact(4).take(6).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let bf16s: Vec<u16> = payload.chunks_exact(2).take(6).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect();
        println!("  i32: {i32s:?}  f32: {f32s:?}  bf16: {bf16s:?}");
    }
}

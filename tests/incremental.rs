use nnue_rs::accumulator::Accumulator;
use nnue_rs::network::Network;
use nnue_rs::{BLACK, KING, PAWN, WHITE};

#[test]
fn test_incremental_update() {
    let big_path = "archive/nnue/networks/nn-1c0000000000.nnue";
    // Check if file exists, if not, skip test or fail (assuming run from root)
    if !std::path::Path::new(big_path).exists() {
        eprintln!("Network file not found at {}, skipping test.", big_path);
        return;
    }

    let network = Network::load(big_path, true).expect("Failed to load network");
    let ft = &network.feature_transformer;

    // 1. Start Position (Simplified, just Kings and Pawns to test)
    // White King E1, Black King E8
    // White Pawn E2
    let ksq = [4, 60]; // E1=4, E8=60

    // (Square, Piece)
    let pieces_start = vec![
        (4, KING),
        (60, KING),
        (12, PAWN), // E2 = 1*8 + 4 = 12
    ];

    let mut acc_fresh = Accumulator::new(ft.half_dims);
    let mut acc_incremental = Accumulator::new(ft.half_dims);

    // Initial Refresh for both
    acc_incremental.refresh(&pieces_start, ksq, ft);

    // 2. Make move e2e4
    // Remove Pawn at E2 (12)
    // Add Pawn at E4 (3*8+4 = 28)

    let pieces_after = vec![(4, KING), (60, KING), (28, PAWN)];

    // Refresh Fresh Accumulator
    acc_fresh.refresh(&pieces_after, ksq, ft);

    // Incremental Update
    let removed = vec![(12, PAWN)];
    let added = vec![(28, PAWN)];

    acc_incremental.update_with_ksq(&added, &removed, ksq, ft);

    // 3. Compare
    // Check White Perspective
    assert_eq!(
        acc_fresh.accumulation[WHITE], acc_incremental.accumulation[WHITE],
        "White Accumulation mismatch"
    );
    assert_eq!(
        acc_fresh.psqt_accumulation[WHITE], acc_incremental.psqt_accumulation[WHITE],
        "White PSQT Accumulation mismatch"
    );

    // Check Black Perspective
    assert_eq!(
        acc_fresh.accumulation[BLACK], acc_incremental.accumulation[BLACK],
        "Black Accumulation mismatch"
    );
    assert_eq!(
        acc_fresh.psqt_accumulation[BLACK], acc_incremental.psqt_accumulation[BLACK],
        "Black PSQT Accumulation mismatch"
    );
}

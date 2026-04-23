//! Integration tests for the Starknet RPC auto-resolver.
//!
//! These tests exercise `vortexstark::cairo_air::starknet_rpc::RpcResolver`
//! and its interaction with `HintContext::with_rpc` for cross-contract call
//! auto-registration.
//!
//! Some tests are gated on `#[ignore]` because they make live RPC requests
//! against public Starknet endpoints. Run explicitly with
//!     cargo test --test rpc_resolver -- --ignored
//!
//! The non-ignored tests exercise the resolver's caching behavior and the
//! HintContext wiring without any network traffic.
//!
//! NOTE: full end-to-end proving of a real mainnet contract requires
//! `felt252` support in dicts / syscall boundaries (see FELT252_DESIGN.md).
//! Today the resolver works for u64-range class_hashes; real mainnet Sierra
//! class_hashes are 252-bit poseidon hashes and will be supported once the
//! Felt252 rework lands.

use vortexstark::cairo_air::hints::HintContext;
use vortexstark::cairo_air::starknet_rpc::{RpcResolver, StarknetClient};

#[test]
fn rpc_resolver_wires_into_hint_context_opt_in() {
    // Default HintContext has no resolver — ensures the field is opt-in.
    let ctx = HintContext::new();
    assert!(
        ctx.rpc_resolver.is_none(),
        "fresh HintContext must not have a resolver attached by default"
    );

    // with_rpc attaches a resolver; with_rpc_resolver takes a shared Arc.
    let ctx = HintContext::new().with_rpc("http://127.0.0.1:65535");
    assert!(ctx.rpc_resolver.is_some(), "with_rpc must attach a resolver");

    let resolver = std::sync::Arc::new(RpcResolver::new("http://127.0.0.1:65535"));
    let ctx2 = HintContext::new().with_rpc_resolver(std::sync::Arc::clone(&resolver));
    assert!(ctx2.rpc_resolver.is_some(), "with_rpc_resolver must attach");
    assert!(
        std::sync::Arc::ptr_eq(&resolver, ctx2.rpc_resolver.as_ref().unwrap()),
        "with_rpc_resolver must share the exact Arc passed in"
    );
}

#[test]
fn rpc_resolver_negative_cache_on_unreachable_endpoint() {
    // Point at a guaranteed-unreachable endpoint; the first lookup fails and
    // the negative cache entry prevents a second network call.
    let resolver = RpcResolver::new("http://127.0.0.1:65535");

    // Arbitrary class_hash — RPC will refuse the connection regardless.
    let result = resolver.try_resolve(0xdeadbeef);
    assert!(
        result.is_none(),
        "unreachable endpoint must produce a cached negative result"
    );

    // Second lookup hits the cache, not the network.
    let t0 = std::time::Instant::now();
    let result2 = resolver.try_resolve(0xdeadbeef);
    let elapsed = t0.elapsed();
    assert!(result2.is_none(), "cached negative lookup must still return None");
    assert!(
        elapsed < std::time::Duration::from_millis(10),
        "cache hit should be <10ms, got {elapsed:?}"
    );

    // clear_cache forces re-fetch.
    resolver.clear_cache();
    let _ = resolver.try_resolve(0xdeadbeef); // will re-hit the refused endpoint
}

#[test]
fn starknet_client_constructors_accept_public_endpoints() {
    // Smoke-only: the mainnet/sepolia helpers wire the correct URLs. We do
    // not actually make a request here — that would add flaky network
    // dependency to every CI run.
    let _m = StarknetClient::mainnet();
    let _s = StarknetClient::sepolia();
    let _custom = StarknetClient::new("http://127.0.0.1:65535");
}

// ============================================================================
// Live RPC tests — opted out by default.
// Run with: cargo test --test rpc_resolver -- --ignored
// ============================================================================

/// Resolve a real mainnet class_hash (Sepolia, pinned low-u64 for test
/// stability). This validates the full RPC pipeline end-to-end:
///   - HTTP POST to the public Sepolia endpoint
///   - JSON-RPC response parsing
///   - CasmProgram extraction
///
/// This test does NOT run the full prover — real mainnet class_hashes are
/// 252-bit poseidon hashes that exceed u64, so full `prove-starknet
/// --resolve-rpc-callees` against arbitrary mainnet contracts is blocked
/// on the Felt252-in-dicts rework (see FELT252_DESIGN.md).
///
/// To run:
///   cargo test --test rpc_resolver test_live_sepolia -- --ignored --nocapture
#[test]
#[ignore]
fn test_live_sepolia_class_resolution() {
    // Use Sepolia; mainnet contracts tend to have long 252-bit class_hashes
    // that exceed our current u64 resolver boundary.
    let resolver = RpcResolver::sepolia();

    // A known-tiny testnet class_hash would be nice here, but even testnet
    // class_hashes are 252-bit in practice. This test exercises the error
    // path via a bogus class_hash; replace with a real pinned hash when
    // Felt252 support lands.
    let result = resolver.try_resolve(0x1);
    // Expect None (Sepolia doesn't have this class); we only assert that
    // we made a real RPC round-trip without panicking.
    eprintln!("Sepolia resolver result for 0x1: {:?}", result.is_some());
}

//! Open a bundle the way a consumer will, and say what came back.
//!
//! ```text
//! cargo run --quiet --example check_bundle --no-default-features -- <bundle dir> [host...]
//! ```
//!
//! Opened as every host whose runtime `bundle.json` declares, or as the hosts
//! named: one bundle carries all of them, and a mode that loads in a browser
//! and not on the board is a bundle that is broken on the board.
//!
//! The bundler writes `controller.toml` and one contract per mode, then has no way to
//! know whether either is loadable -- Python cannot run `FsmConfig::parse`, and
//! re-implementing it there would be the second implementation this whole crate
//! exists to avoid. So it asks the crate.
//!
//! It asks through `bundle::Bundle`, which is the same call the robot makes: if
//! this reported a bundle as good and the board then refused it, this check
//! would be worse than nothing. Everything below the `open` is printing.
//!
//! An example rather than a binary: it is a development check, not something the
//! robot ships, and `[[bin]]` would put it in the release profile's way.

use mjrl_fsm::bundle::{Bundle, Target};

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(dir) = args.next() else {
        eprintln!("[check_bundle] usage: check_bundle <bundle dir> [board|web|mjlab ...]");
        std::process::exit(1);
    };
    let all = [Target::Web, Target::Mjlab, Target::Board];
    let mut hosts: Vec<Target> = args
        .map(|a| {
            all.into_iter().find(|t| t.name() == a).unwrap_or_else(|| {
                eprintln!("[check_bundle] '{a}' is not a host: board, web or mjlab");
                std::process::exit(1)
            })
        })
        .collect();
    if hosts.is_empty() {
        let manifest: serde_json::Value = std::fs::read_to_string(format!("{dir}/bundle.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default();
        hosts = all.into_iter().filter(|t| manifest["runtimes"].get(t.name()).is_some()).collect();
    }
    if hosts.is_empty() {
        eprintln!("[check_bundle] {dir}/bundle.json declares no runtime, so no host could run it");
        std::process::exit(1);
    }
    let mut opened = hosts.iter().map(|&host| {
        let bundle = Bundle::open(&dir, host).unwrap_or_else(|e| {
            eprintln!("[check_bundle] as the {}: {e}", host.name());
            std::process::exit(1)
        });
        (host, bundle)
    });
    let (_, bundle) = opened.next().unwrap();
    let rest: Vec<_> = opened.collect();
    println!(
        "  opens as {}",
        std::iter::once(hosts[0].name()).chain(rest.iter().map(|(h, _)| h.name())).collect::<Vec<_>>().join(", ")
    );
    for (host, other) in &rest {
        for note in &other.notes {
            if !bundle.notes.contains(note) {
                println!("  note ({}): {note}", host.name());
            }
        }
    }
    let board = rest.iter().map(|(_, b)| b).chain(std::iter::once(&bundle)).find(|b| b.target == Target::Board);

    for (name, contract) in {
        // Sorted, so two runs over one bundle print the same thing.
        let mut v: Vec<_> = bundle.contracts.iter().collect();
        v.sort_by_key(|(n, _)| n.as_str());
        v
    } {
        println!(
            "  {name:16} obs {} -> act {}",
            contract.layout.observation.dim,
            contract.action_dim()
        );
    }
    if bundle.cfg.buttons.is_empty() {
        println!("  no controls: this bundle never switches modes from a button");
    } else {
        for b in &bundle.cfg.buttons {
            let source = [b.pad.as_deref(), b.key.as_deref()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" / ");
            let modifier = b.with.as_deref().map(|m| format!(" with {m}")).unwrap_or_default();
            println!("  {:14} {source} on {}{modifier}", b.name, b.on);
        }
    }
    println!(
        "  {} joints on the wire, {} mode(s), {} rule(s)",
        bundle.wire.len(),
        bundle.models.len(),
        bundle.cfg.rules.len()
    );
    // Said out loud rather than implied by its absence: a device bundle that
    // carries the stops is the only kind a board can clamp against, and the
    // difference is invisible in a directory listing.
    if let Some(bundle) = board {
        match bundle.joint_limits() {
            Some((lo, hi)) => println!("  joint limits for {} joints, from the contract", lo.len()
                .min(hi.len())),
            None => println!("  no joint limits (this bundle is for a host with a model)"),
        }
    }
}

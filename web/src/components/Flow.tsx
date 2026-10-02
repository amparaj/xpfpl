// The system as a flow diagram: data in at the top, a team out at the bottom. Built from HTML boxes
// rather than an image so it reflows: steps that run side by side sit in a row on a wide screen and
// stack on a phone. Each box links to its section of the technical report.

import type { ReactNode } from "react";
import { useSite } from "../site";

interface Node { title: string; text?: ReactNode; section?: string; kind?: "source" | "model" | "key" | "out" }

function Box({ node, detailed }: { node: Node; detailed: boolean }) {
  const body = (
    <>
      <span className="flow-title">{node.title}</span>
      {detailed && node.text && <span className="flow-text">{node.text}</span>}
    </>
  );
  const cls = `flow-node${node.kind ? ` flow-${node.kind}` : ""}`;
  return node.section
    ? <a className={cls} href={`#about/report/${node.section}`}>{body}</a>
    : <div className={cls}>{body}</div>;
}

/** One stage: its label on the left (above, on a phone) and its boxes, joined into the next stage by an arrow. */
function Stage({ label, nodes, detailed, join, note }: {
  label: string; nodes: Node[]; detailed: boolean; join?: string; note?: ReactNode;
}) {
  return (
    <div className="flow-stage">
      <div className="flow-label">{label}</div>
      <div className="flow-row" style={{ ["--n" as string]: nodes.length }}>
        {nodes.map((n) => <Box key={n.title} node={n} detailed={detailed} />)}
      </div>
      {note && <div className="flow-note">{note}</div>}
      {join !== undefined && <div className="flow-arrow" aria-hidden><span>{join}</span></div>}
    </div>
  );
}

/** The whole system. `detailed` adds a line to each box (the technical report); without it, titles only (About). */
export function Flow({ detailed = false, horizon = 3 }: { detailed?: boolean; horizon?: number }) {
  const site = useSite();
  const ensemble = site.meta.model === "ensemble";
  const models: Node[] = ensemble
    ? [
        { title: "Neural network", text: "PyTorch MLP, 128-64, on every feature", section: "models", kind: "model" },
        { title: "Gradient boosting", text: "LightGBM trees on the same features", section: "models", kind: "model" },
        { title: "Minutes model", text: "PyTorch: chance of 0 / 1-59 / 60+ minutes, then points given each", section: "models", kind: "model" },
      ]
    : [{ title: site.meta.model, text: site.meta.model_description, section: "models", kind: "model" }];
  return (
    <figure className={`flow${detailed ? " flow-detailed" : ""}`} aria-label="How xP-FPL works, from data to a team">
      <Stage label="Data" detailed={detailed} join="archived, then turned into features" nodes={[
        { title: "FPL", text: "this season, live from FPL's API", section: "data", kind: "source" },
        { title: "Past seasons", text: "every match since 2016-17", section: "data", kind: "source" },
        { title: "Betting odds", text: "Polymarket and bookmakers", section: "data", kind: "source" },
        { title: "Cups and Europe", text: "midweek fixtures and minutes", section: "data", kind: "source" },
      ]} />
      <Stage label="Features" detailed={detailed} join="for each player and fixture" nodes={[
        { title: "Minutes and role", text: "starts, minutes, first choice at his club", section: "inputs" },
        { title: "Underlying numbers", text: "rolling form, xG and xA per 90", section: "inputs" },
        { title: "Fixture", text: "club ratings and market goal expectations", section: "inputs" },
        { title: "Crowd", text: "transfers in and out before the deadline", section: "inputs" },
      ]} />
      <Stage label="Forecast" detailed={detailed} join={ensemble ? "averaged into one forecast" : "one forecast per player"} nodes={models} />
      <Stage label="Adjust" detailed={detailed} join="" nodes={[
        { title: "Injury flags", text: "FPL's chance of playing, recovering over the weeks", section: "models" },
        { title: "Midweek factor", text: "his minutes in the cup or European match", section: "models" },
        { title: "Ruled out", text: "goalscorer odds of 6% or less", section: "models" },
      ]} />
      <Stage label="xP" detailed={detailed} join="used two ways" nodes={[
        { title: `Expected points for every player, next ${horizon} gameweeks`, text: "saved before every deadline", section: "accuracy", kind: "key" },
      ]} />
      <Stage label="Use" detailed={detailed} join="" nodes={[
        { title: "Monte Carlo", text: "each gameweek played thousands of times: ranges, captain and chip odds", section: "simulation" },
        { title: "Optimiser", text: "integer program: squad, transfers, XI, captain, chips", section: "selection" },
      ]} />
      <Stage label="Out" detailed={detailed} nodes={[
        { title: "This Model's Team", text: "a paper FPL team, decided before every deadline", kind: "out" },
        { title: "Scorecard", text: "every forecast scored once the gameweek is played", section: "validation", kind: "out" },
      ]} note={<>↺ The scorecard and replayed seasons feed back into which model and settings are used.</>} />
    </figure>
  );
}

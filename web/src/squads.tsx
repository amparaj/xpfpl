// The two teams the player tables can be cut down to: My Team (myteam.json; before the deadline only
// with the secret word, see seal.ts) and This Model's Team (modelteam.json, the squad of its latest decision).

import { useEffect, useMemo, useState } from "react";
import { Segmented } from "./components/ui";
import { rows, type Forecast, type ModelTeam, type MyTeam, type MyTeamPrivate } from "./data";
import { rememberWord, rememberedWord, unseal } from "./seal";
import { useData } from "./site";

/** My team: the data's own after the deadline, else what the secret word unlocked (remembered in this browser). */
export function useMyTeam(data: MyTeam | null | undefined) {
  const [unlocked, setUnlocked] = useState<MyTeamPrivate | null>(null);
  const [state, setState] = useState<"idle" | "busy" | "wrong">("idle");
  const sealed = data?.sealed ?? null;

  const unlock = async (word: string, remember: boolean) => {
    if (!sealed || !word.trim()) return;
    setState("busy");
    const opened = await unseal<MyTeamPrivate>(sealed, word);
    setUnlocked(opened);
    setState(opened ? "idle" : "wrong");
    if (opened && remember) rememberWord(word);
    if (!opened && !remember) rememberWord(null);          // a remembered word that no longer works
  };
  useEffect(() => {
    const word = rememberedWord();
    if (sealed && word) void unlock(word, false);
  }, [sealed]); // eslint-disable-line react-hooks/exhaustive-deps

  const lock = () => { rememberWord(null); setUnlocked(null); setState("idle"); };
  return { team: data?.private ?? unlocked, unlocked: !!unlocked, state, unlock, lock };
}

export type Squad = "all" | "mine" | "model";

/** Both squads as sets of player ids. `mine` is null while My Team is locked or there isn't one. */
export function useSquads() {
  const myData = useData<MyTeam>("myteam.json");
  const modelTeam = useData<ModelTeam>("modelteam.json");
  const { team } = useMyTeam(myData);
  const mine = useMemo(() => (team ? new Set(rows<Forecast>(team.players).map((r) => r.element)) : null), [team]);
  const model = useMemo(() => {
    const weeks = modelTeam?.gameweeks ?? [];
    const latest = modelTeam?.next ?? weeks[weeks.length - 1];
    return new Set(latest?.squad ?? []);
  }, [modelTeam]);
  /** Why My Team can't be shown: no team exported, or one that's sealed and not unlocked here. */
  const why = mine ? null : myData?.sealed ? "locked" : "none";
  return { mine, model, why };
}

/** Whether player `id` is in the chosen squad. */
export function inSquad(squads: ReturnType<typeof useSquads>, squad: Squad, id: number): boolean {
  if (squad === "all") return true;
  return (squad === "mine" ? squads.mine : squads.model)?.has(id) ?? false;
}

/** All players / My Team / This model's team, with a line saying why My Team is empty when it is. */
export function SquadFilter({ value, onChange, squads }: {
  value: Squad; onChange: (v: Squad) => void; squads: ReturnType<typeof useSquads>;
}) {
  return (
    <>
      <Segmented label="Team" value={value} onChange={onChange}
                 options={[{ value: "all", label: "All players" }, { value: "mine", label: "My Team" }, { value: "model", label: "This model's team" }]} />
      {value === "mine" && squads.why && (
        <span className="muted" style={{ fontSize: 13 }}>
          {squads.why === "locked"
            ? <>My Team is locked until the deadline: open it with the secret word on <a href="#my-team">My Team</a>.</>
            : <>No team saved for this gameweek yet.</>}
        </span>
      )}
    </>
  );
}

/** "MT" / "Mine" tags after a player's name. */
export function SquadTags({ id, squads }: { id: number; squads: ReturnType<typeof useSquads> }) {
  return (
    <>
      {squads.mine?.has(id) && <> <span className="tag" title="In My Team">Mine</span></>}
      {squads.model.has(id) && <> <span className="tag" title="In this model's team">MT</span></>}
    </>
  );
}

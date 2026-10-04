export const reportBins = [
  { label: '0', colour: '#edf1e9', minimum: 0 },
  { label: '1', colour: '#b4d4be', minimum: 1 },
  { label: '2–4', colour: '#74af91', minimum: 2 },
  { label: '5–9', colour: '#347d64', minimum: 5 },
  { label: '10+', colour: '#174d40', minimum: 10 },
];

export function reportColour(count: number) {
  return [...reportBins].reverse().find(bin => count >= bin.minimum)!.colour;
}

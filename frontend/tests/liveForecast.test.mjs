import test from 'node:test';
import assert from 'node:assert/strict';
import { parseForecast } from '../src/liveForecast.ts';

const now=Date.parse('2026-10-04T13:20:00Z');
function payload(timestamp=now){
  const hour=Math.floor(timestamp/3600000)*3600;
  return {hourly_units:{rain:'mm',showers:'mm'},current:{temperature_2m:12},hourly:{time:Array.from({length:38},(_,i)=>hour+i*3600),rain:Array(38).fill(.2),showers:Array(38).fill(.3),precipitation_probability:Array(38).fill(50)}};
}
test('selects 36 upcoming complete intervals and sums rain plus showers',()=>{
  const result=parseForecast(payload(),500,now);
  assert.equal(result.forecastStart,'2026-10-04T14:00:00.000Z');
  assert.equal(result.forecastEnd,'2026-10-06T02:00:00.000Z');
  assert.equal(result.forecast.length,36);assert.equal(result.totalRainMm,18);
  assert.equal(result.forecast[0].time,'2026-10-04T15:00:00.000Z');
});
test('rejects missing rainfall instead of treating it as dry',()=>{
  const data=payload();data.hourly.rain[2]=null;
  assert.throws(()=>parseForecast(data,500,now),/missing/);
});
test('rejects duplicated hours and incorrect units',()=>{
  const data=payload();data.hourly.time[3]=data.hourly.time[2];
  assert.throws(()=>parseForecast(data,500,now),/continuous/);
  const units=payload();units.hourly_units.rain='inch';
  assert.throws(()=>parseForecast(units,500,now),/units/);
});
test('uses continuous UTC hours through Vancouver fall DST change',()=>{
  const dst=Date.parse('2026-11-01T08:20:00Z');
  const result=parseForecast(payload(dst),500,dst);
  assert.equal(Date.parse(result.forecastEnd)-Date.parse(result.forecastStart),36*3600000);
  assert.equal(result.forecast.length,36);
});

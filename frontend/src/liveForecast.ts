export type Forecast = {gridSize:number;generatedAt:string;forecastStart:string;forecastEnd:string;forecast:{hour:number;time:string;rainMm:number;probability:number|null}[];totalRainMm:number;temperature:number|null};
export const forecastUrl='https://api.open-meteo.com/v1/forecast?latitude=49.2827&longitude=-123.1207&hourly=rain,showers,precipitation_probability&current=temperature_2m&forecast_hours=38&timezone=GMT&timeformat=unixtime';
type Payload={hourly:{time:number[];rain:(number|null)[];showers:(number|null)[];precipitation_probability?:(number|null)[]};hourly_units:{rain:string;showers:string};current?:{temperature_2m?:number|null}};
export function parseForecast(payload:Payload,size:number,now=Date.now()):Forecast {
  const start=Math.ceil(now/3600000)*3600000,end=start+36*3600000;
  if(payload.hourly_units?.rain!=='mm'||payload.hourly_units?.showers!=='mm'||!Array.isArray(payload.hourly?.time))throw new Error('Unexpected forecast units or timestamps.');
  const forecast:Forecast['forecast']=[];
  payload.hourly.time.forEach((seconds,index)=>{
    const timestamp=seconds*1000;
    // Hourly rainfall timestamps mark the END of the preceding interval.
    if(timestamp<=start||timestamp>end)return;
    const rain=payload.hourly.rain[index],showers=payload.hourly.showers[index],chance=payload.hourly.precipitation_probability?.[index];
    if(typeof rain!=='number'||typeof showers!=='number'||!Number.isFinite(rain+showers)||rain<0||showers<0)throw new Error('Some forecast hours are missing. Please try again.');
    forecast.push({hour:forecast.length,time:new Date(timestamp).toISOString(),rainMm:Math.round((rain+showers)*100)/100,probability:typeof chance==='number'&&chance>=0&&chance<=100?chance:null});
  });
  if(forecast.length!==36||forecast.some((point,index)=>Date.parse(point.time)!==start+(index+1)*3600000))throw new Error('A continuous 36-hour forecast is unavailable. Please try again.');
  const temperature=payload.current?.temperature_2m;
  return {gridSize:size,generatedAt:new Date(now).toISOString(),forecastStart:new Date(start).toISOString(),forecastEnd:new Date(end).toISOString(),forecast,totalRainMm:Math.round(forecast.reduce((sum,point)=>sum+point.rainMm,0)*100)/100,temperature:typeof temperature==='number'&&Number.isFinite(temperature)?temperature:null};
}
export async function requestLiveForecast(size:number,signal:AbortSignal):Promise<Forecast>{
  const response=await fetch(forecastUrl,{signal:AbortSignal.any([signal,AbortSignal.timeout(20000)])});
  if(!response.ok)throw new Error(`Weather service unavailable (${response.status}). Please try again.`);
  return parseForecast(await response.json(),size);
}

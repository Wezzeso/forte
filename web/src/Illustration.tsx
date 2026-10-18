import { MagnifyingGlassIcon } from '@phosphor-icons/react'

/** Original geometric artwork, sharing the site's magenta, paper and ink palette. */
export default function Illustration({ kind, className = '' }: {
  kind: 'upload' | 'inspect' | 'analyze' | 'decide' | 'access' | 'clear' | 'alert'
  className?: string
}) {
  return <svg className={`illustration illustration-${kind} ${className}`} viewBox="0 0 200 180" fill="none" aria-hidden="true" focusable="false">
    {kind === 'analyze' && <>
      <path d="m23 118 20-52 27 11-13 32 28-15 16 26-56 29Z" fill="#E9CAD8" />
      <g transform="rotate(-8 106 89)">
        <path d="M69 24h72v130H69Z" fill="#A31551" />
        <path d="M79 34h52v110H79Z" fill="#F2F2F2" />
        <path d="M73 40h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2m60-102h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2m-2 17h2" stroke="#F2F2F2" strokeWidth="3" />
        <path d="m89 60 9-15 10 15 12-10v25H89Z" fill="#E9CAD8" />
        <path d="m89 118 9-15 10 15 12-10v25H89Z" fill="#E9CAD8" />
        <path d="M79 88h52" stroke="#A31551" strokeWidth="3" />
      </g>
      <g className="analysis-lens">
        <circle cx="105" cy="82" r="27" fill="#F2F2F2" fillOpacity=".92" />
        <path d="m91 88 9-13 10 9 9-12" stroke="#A31551" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        <MagnifyingGlassIcon x={64} y={41} size={94} weight="bold" color="#161116" />
        <path d="M88 73a19 19 0 0 1 13-9" stroke="white" strokeWidth="4" strokeLinecap="round" />
      </g>
      <g className="analysis-spark analysis-spark-first"><path d="m158 27 4 11 12 4-12 4-4 12-4-12-11-4 11-4Z" fill="#161116" /></g>
      <g className="analysis-spark analysis-spark-second"><path d="m45 28 3 8 9 3-9 3-3 9-3-9-8-3 8-3Z" fill="#A31551" /></g>
      <circle cx="151" cy="146" r="6" fill="#A31551" />
    </>}
    {kind === 'upload' && <>
      <path d="M24 124 42 79 65 104 89 51 111 79 142 26 164 43 133 98 111 77 86 137 62 113 46 151Z" fill="#E9CAD8" />
      <g transform="rotate(9 108 94)">
        <path d="M75 25H142V150H75Z" fill="#A31551" />
        <path d="M84 36H133V139H84Z" fill="#F2F2F2" />
        <path d="m102 74 23 14-23 14Z" fill="#A31551" />
        <path d="M79 43h2m-2 16h2m-2 16h2m-2 16h2m-2 16h2m-2 16h2m55-80h2m-2 16h2m-2 16h2m-2 16h2m-2 16h2m-2 16h2" stroke="#F2F2F2" strokeWidth="3" />
      </g>
      <path d="M38 61c-8-19 2-34 21-32M49 19l12 10-12 10" stroke="#161116" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m156 110 7 13 15 3-12 10 2 16-14-8-14 6 3-15-10-12 16-1Z" fill="#161116" />
      <circle cx="46" cy="156" r="7" fill="#A31551" />
    </>}
    {kind === 'inspect' && <>
      <path d="m22 90 31-54 24 14-21 38 32-15 14 26-55 29Z" fill="#E9CAD8" />
      <path d="M80 24h69v125H80Z" transform="rotate(12 114 86)" fill="#A31551" />
      <path d="M92 40h44v88H92Z" transform="rotate(12 114 86)" fill="#F2F2F2" />
      <path d="M40 100c25-43 71-43 98 0-27 43-73 43-98 0Z" fill="#161116" />
      <path d="M51 100c20-27 49-27 75 0-26 27-55 27-75 0Z" fill="#F2F2F2" />
      <circle cx="89" cy="100" r="19" fill="#A31551" />
      <circle cx="93" cy="95" r="6" fill="#F2F2F2" />
      <path d="m137 133 16 18M165 50v18m-9-9h18" stroke="#161116" strokeWidth="5" strokeLinecap="round" />
      <circle cx="55" cy="149" r="7" fill="#A31551" />
    </>}
    {(kind === 'decide' || kind === 'clear') && <>
      <path d="m35 114 20-66 27 8-12 40 31-23 20 28-59 43Z" fill="#E9CAD8" />
      <path d="m100 24 10 18 20-5-4 21 18 11-16 14 9 20-22 1-5 21-17-13-17 13-6-21-21-1 9-20-16-14 18-11-4-21 20 5 10-18Z" fill="#A31551" />
      <path d="m76 73 15 16 27-32" stroke="#F2F2F2" strokeWidth="9" strokeLinecap="square" />
      <path d="M113 142c21 3 40-10 42-28m-11 5 12-8 7 14" stroke="#161116" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m35 32 6 9 11 1-7 8 2 11-11-5-9 6 1-11-9-7 11-2Z" fill="#161116" />
    </>}
    {kind === 'access' && <>
      <path d="m33 126 22-70 28 9-10 34 32-18 14 26-57 35Z" fill="#E9CAD8" />
      <path d="M148 64a32 32 0 1 0-58 19l-43 44 19 19 12-12-2-13 14 1 20-21a32 32 0 0 0 38-37Z" fill="#A31551" />
      <circle cx="121" cy="63" r="11" fill="#F2F2F2" />
      <path d="m157 108 7 12 14 2-10 10 1 14-12-7-13 5 3-14-9-10 14-1Z" fill="#161116" />
      <path d="m49 32 7 11m18-18-1 13M33 51l12 2" stroke="#161116" strokeWidth="4" strokeLinecap="round" />
    </>}
    {kind === 'alert' && <>
      <path d="m23 121 29-72 27 11-15 38 30-16 14 27-61 34Z" fill="#E9CAD8" />
      <path d="m112 24 9 18 21-4-3 21 18 12-16 14 8 20-22 1-5 21-17-13-18 13-5-21-21-1 8-20-16-14 18-12-3-21 21 4 9-18Z" fill="#A31551" />
      <path d="M108 54v28" stroke="#F2F2F2" strokeWidth="9" />
      <circle cx="108" cy="96" r="5" fill="#F2F2F2" />
      <path d="M129 146c17-1 29-13 29-27m-10 5 10-8 8 10" stroke="#161116" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
    </>}
  </svg>
}

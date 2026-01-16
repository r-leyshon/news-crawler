// Shared region constants and utilities

export const availableRegions = [
  { code: "uk-en", name: "United Kingdom", flag: "🇬🇧" },
  { code: "us-en", name: "United States", flag: "🇺🇸" },
  { code: "de-de", name: "Germany", flag: "🇩🇪" },
  { code: "fr-fr", name: "France", flag: "🇫🇷" },
  { code: "it-it", name: "Italy", flag: "🇮🇹" },
  { code: "es-es", name: "Spain", flag: "🇪🇸" },
  { code: "nl-nl", name: "Netherlands", flag: "🇳🇱" },
  { code: "ca-en", name: "Canada", flag: "🇨🇦" },
  { code: "au-en", name: "Australia", flag: "🇦🇺" },
  { code: "in-en", name: "India", flag: "🇮🇳" }
]

export interface RegionInfo {
  flag: string
  name: string
}

// Helper function to get region display info
export const getRegionInfo = (regionCode?: string): RegionInfo | null => {
  if (!regionCode) return null
  const region = availableRegions.find(r => r.code === regionCode)
  return region ? { flag: region.flag, name: region.name } : { flag: "🌐", name: regionCode }
} 
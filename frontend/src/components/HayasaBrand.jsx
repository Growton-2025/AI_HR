import React from 'react'

export const GROWTON_LOGO_URL = 'https://cdn.prod.website-files.com/65e41b0d7632a225ef3abc4e/693509bb8da9f3b5e10554be_LOGO.svg'

const sizeMap = {
  hero: {
    productSize: 29,
    logoWidth: 96,
    logoHeight: 25,
    dotPad: '3px 8px',
  },
  sidebar: {
    productSize: 15,
    logoWidth: 52,
    logoHeight: 14,
    dotPad: '1px 5px',
  },
  compact: {
    productSize: 13,
    logoWidth: 42,
    logoHeight: 11,
    dotPad: '1px 5px',
  },
}

export default function HayasaBrand({
  size = 'sidebar',
  tone = 'dark',
  iconOnly = false,
  showGrowton = true,
}) {
  const config = sizeMap[size] || sizeMap.sidebar
  const isDark = tone === 'dark'
  const productColor = isDark ? '#f8fafc' : '#111827'
  const accentBackground = isDark ? 'rgba(244,190,119,0.18)' : '#f4e3cf'
  const accentColor = isDark ? '#f7d1a1' : '#8b5e34'

  if (iconOnly) {
    return (
      <div
        aria-label="Growton Hayasa.ai"
        style={{
          width: 34,
          height: 34,
          borderRadius: 11,
          border: `1px solid ${isDark ? 'rgba(255,255,255,0.14)' : '#dbe3ee'}`,
          background: isDark ? 'rgba(255,255,255,0.06)' : '#fff',
          color: productColor,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 18,
          fontWeight: 900,
          letterSpacing: 0,
          flexShrink: 0,
        }}
      >
        H
      </div>
    )
  }

  return (
    <div
      aria-label="Growton Hayasa.ai"
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        flexWrap: size === 'hero' ? 'wrap' : 'nowrap',
        gap: size === 'hero' ? 14 : size === 'compact' ? 6 : 10,
        minWidth: 0,
      }}
    >
      {showGrowton && (
        <>
          <img
            src={GROWTON_LOGO_URL}
            alt="Growton"
            style={{
              width: config.logoWidth,
              height: config.logoHeight,
              objectFit: 'contain',
              filter: isDark ? 'brightness(0) invert(1)' : 'brightness(0)',
              opacity: isDark ? 0.98 : 0.92,
              flexShrink: 0,
            }}
          />
          <span
            aria-hidden="true"
            style={{
              width: 1,
              height: size === 'hero' ? 29 : size === 'compact' ? 15 : 20,
              background: isDark ? 'rgba(255,255,255,0.18)' : '#dbe3ee',
              flexShrink: 0,
            }}
          />
        </>
      )}
      <div
        style={{
          display: 'flex',
          alignItems: 'baseline',
          gap: size === 'compact' ? 4 : 6,
          color: productColor,
          whiteSpace: 'nowrap',
          letterSpacing: 0,
        }}
      >
        <span style={{ fontSize: config.productSize, fontWeight: 900, lineHeight: 1 }}>
          Hayasa
        </span>
        <span
          style={{
            padding: config.dotPad,
            borderRadius: 999,
            background: accentBackground,
            color: accentColor,
            fontSize: Math.max(config.productSize - 7, 10),
            fontWeight: 900,
            lineHeight: 1.15,
          }}
        >
          .ai
        </span>
      </div>
    </div>
  )
}

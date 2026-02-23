/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'neon-blue':  '#00f2ff',
        'neon-red':   '#ff2a2a',
        'neon-green': '#00ff88',
        'dark-bg':    '#020617',
        'dark-card':  '#0f1c3f',
      },
      backgroundImage: {
        'radial-dark': 'radial-gradient(circle at center, #0f1c3f 0%, #020617 100%)',
      },
      backdropBlur: {
        xs: '4px',
      },
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['Courier New', 'monospace'],
      },
      keyframes: {
        pulse: {
          '0%':   { boxShadow: '0 0 0 0 rgba(0, 242, 255, 0.4)' },
          '70%':  { boxShadow: '0 0 0 10px rgba(0, 242, 255, 0)' },
          '100%': { boxShadow: '0 0 0 0 rgba(0, 242, 255, 0)' },
        },
        fadeInUp: {
          from: { opacity: '0', transform: 'translateY(20px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        'pulse-neon': 'pulse 2s infinite',
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
      },
    },
  },
  plugins: [],
}

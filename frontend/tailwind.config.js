/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#fef2f1",
          100: "#fde3e0",
          400: "#ec6a5c",
          500: "#e0402d",
          600: "#c72f1e",
          700: "#a12419",
        },
        ink: {
          900: "#12162b",
          800: "#1c2140",
          600: "#454b6b",
          400: "#7c8199",
        },
      },
      fontFamily: {
        sans: ["Sora", "ui-sans-serif", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

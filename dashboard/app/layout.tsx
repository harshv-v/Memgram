import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Memgram Dashboard",
  description: "View and edit your agent's memory.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

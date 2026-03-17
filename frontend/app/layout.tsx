import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "STEP Study Assistant | RAG Textbook Chatbot",
  description: "AI-powered study assistant with your entire STEP textbook indexed. Ask questions, get answers with precise page and chapter citations.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

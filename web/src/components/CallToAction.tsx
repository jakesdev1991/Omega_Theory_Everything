import Link from "next/link";

interface Props {
  label: string;
  href: string;
  variant?: "solid" | "ghost";
}

export function CallToAction({ label, href, variant = "solid" }: Props) {
  if (variant === "ghost") {
    return (
      <Link
        href={href}
        className="btn-ghost"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          padding: "10px 20px",
          borderRadius: "8px",
          border: "1px solid var(--color-border-strong)",
          color: "var(--color-foreground)",
          fontSize: "14px",
          fontWeight: 500,
          textDecoration: "none",
        }}
      >
        {label}
      </Link>
    );
  }

  return (
    <Link
      href={href}
      className="btn-solid"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: "8px",
        padding: "12px 24px",
        borderRadius: "10px",
        background: "linear-gradient(135deg, var(--color-accent), #a86a36)",
        color: "#07080c",
        fontSize: "14px",
        fontWeight: 600,
        textDecoration: "none",
        letterSpacing: "0.01em",
        boxShadow: "0 6px 24px rgba(192,132,87,0.25)",
      }}
    >
      {label}
    </Link>
  );
}

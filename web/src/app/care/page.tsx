import type { Metadata } from "next";

import { CareSocialPrototype } from "@/components/CareSocialPrototype";

export const metadata: Metadata = {
  title: "C.A.R.E. Social Prototype",
  description:
    "A local, valueless prototype for the C.A.R.E. social layer: circles, care requests, reach, hardship solidarity, and user-controlled boundaries.",
};

export default function CarePage() {
  return <CareSocialPrototype />;
}

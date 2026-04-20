# LinkerAssert.MachOAddendRewriter

```xml
<ItemGroup>
  <PackageReference Include="LinkerAssert.MachOAddendRewriter" Version="0.1.0" />
</ItemGroup>
```

The package adds an MSBuild target that runs after `IlcCompile` and rewrites
each `ManagedBinary` item with a non-empty `IlcOutputFile` metadata value.

Disable it with:

```xml
<PropertyGroup>
  <LinkerAssertMachOAddendRewriterEnabled>false</LinkerAssertMachOAddendRewriterEnabled>
</PropertyGroup>
```
